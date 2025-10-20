# Variables
variable "services" {
  description = "Comma-separated list of service names"
  type        = list(string)
  default     = ["AirbnbSearch","WeatherSearch"] # "MflixSearch"
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "mcp-search-app"
}

variable "ecr_repository" {
  description = "ECR repository URL"
  type        = string
  default     = "<<AWS ACCOUNTID>>.dkr.ecr.<<REGION>>.amazonaws.com/mongodb-dynamic-mcp"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "certificate_arn" {
  description = "AWS Certificate Manager certificate ARN for SSL"
  type        = string
  default     = "arn:aws:acm:<<REGION>>:<<AWS ACCOUNTID>>:certificate/YOUR_CERT_ARN_HERE"
}

variable "K8service_account" {
  description = "The Kubernetes service account to use for the deployments"
  type        = string
  default     = "mcp-mcp-sa"
}

variable "mongo_creds" {
  description = "MongoDB credentials identifier"
  type        = string
  default     = "demo1"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "your-eks-cluster-name"
}

variable "iam_role_arn" {
  description = "IAM role ARN for service account (IRSA)"
  type        = string
  default     = "arn:aws:iam::<<YOUR AWS ACCOUNT>>:role/<<ROLENAME>>"
}

# Local values
locals {
  service_map = {
    for service in var.services : lower(service) => {
      name = service
      lower_name = lower(service)
    }
  }
}

# Providers
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
}

# Data source to get EKS cluster information
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

# Configure Kubernetes Provider for EKS
provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}
provider "helm" {
   kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}


# Namespace
resource "kubernetes_namespace" "mcp_search_namespace" {
  metadata {
    name = var.namespace
  }
}

# Service Account
resource "kubernetes_service_account" "mcp_search_service_account" {
  metadata {
    name      = var.K8service_account
    namespace =  var.namespace
    annotations = {
      "eks.amazonaws.com/role-arn" = var.iam_role_arn
    }
  }
}

# Deployments
resource "kubernetes_deployment" "mcp_search_deployments" {
  for_each = local.service_map

  metadata {
    name      = "${each.value.lower_name}-mcp-app"
    namespace =  var.namespace
    labels = {
      app = "${each.value.lower_name}-mcp-app"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "${each.value.lower_name}-mcp-app"
      }
    }

    template {
      metadata {
        labels = {
          app = "${each.value.lower_name}-mcp-app"
        }
      }

      spec {
        service_account_name = var.K8service_account

        container {
          name  = "${each.value.lower_name}-mcp-container"
          image = "${var.ecr_repository}:${var.image_tag}"

          env {
            name  = "MCP_TOOL_NAME"
            value = each.value.name
          }

          env {
            name  = "MONGO_CREDS"
            value = var.mongo_creds
          }

          port {
            container_port = 8000
          }

          resources {
            limits = {
              memory = "512Mi"
              cpu    = "500m"
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_service_account.mcp_search_service_account
  ]
}

# Services
resource "kubernetes_service" "mcp_search_services" {
  for_each = local.service_map

  metadata {
    namespace =  var.namespace
    name      = "${each.value.lower_name}-mcp-service"
    labels = {
      app = "${each.value.lower_name}-mcp-app"
    }
    annotations = {
      "alb.ingress.kubernetes.io/healthcheck-path" = "/${each.value.name}/health"
    }
  }

  spec {
    type = "NodePort"

    selector = {
      app = "${each.value.lower_name}-mcp-app"
    }

    port {
      port        = 8000
      target_port = 8000
    }
  }
}

# High Priority Ingress for Service-Specific Paths
resource "kubernetes_ingress_v1" "mcp_search_ingress_services" {
  metadata {
    name      = "mcp-ingress-services"
    namespace =  var.namespace
    labels = {
      app = "mcp-app"
    }
    annotations = {
      "kubernetes.io/ingress.class"                                = "alb"
      "alb.ingress.kubernetes.io/scheme"                           = "internet-facing"
      "alb.ingress.kubernetes.io/target-type"                      = "ip"
      "alb.ingress.kubernetes.io/group.name"                       = "mcp-group"
      "alb.ingress.kubernetes.io/group.order"                      = "10"
      "alb.ingress.kubernetes.io/healthcheck-protocol"             = "HTTP"
      "alb.ingress.kubernetes.io/healthcheck-port"                 = "traffic-port"
      "alb.ingress.kubernetes.io/healthcheck-interval-seconds"     = "15"
      "alb.ingress.kubernetes.io/healthcheck-timeout-seconds"      = "5"
      "alb.ingress.kubernetes.io/success-codes"                    = "200"
      "alb.ingress.kubernetes.io/healthy-threshold-count"          = "2"
      "alb.ingress.kubernetes.io/unhealthy-threshold-count"        = "2"
      # SSL Configuration
      "alb.ingress.kubernetes.io/listen-ports"    = "[{\"HTTP\": 80}, {\"HTTPS\":443}]"
      "alb.ingress.kubernetes.io/certificate-arn" = var.certificate_arn
      "alb.ingress.kubernetes.io/ssl-policy"      = "ELBSecurityPolicy-TLS13-1-2-Res-2021-06"
    }
  }

  spec {
    rule {
      http {
        dynamic "path" {
          for_each = local.service_map
          content {
            path      = "/${path.value.name}"
            path_type = "Prefix"
            backend {
              service {
                name = "${path.value.lower_name}-mcp-service"
                port {
                  number = 8000
                }
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_service.mcp_search_services
  ]
}

# Lower Priority Ingress for Catch-All Path
resource "kubernetes_ingress_v1" "mcp_search_ingress_default" {
  metadata {
    name      = "mcp-ingress-default"
    namespace =  var.namespace
    labels = {
      app = "mcp-app"
    }
    annotations = {
      "kubernetes.io/ingress.class"                                = "alb"
      "alb.ingress.kubernetes.io/scheme"                           = "internet-facing"
      "alb.ingress.kubernetes.io/target-type"                      = "ip"
      "alb.ingress.kubernetes.io/group.name"                       = "mcp-group"
      "alb.ingress.kubernetes.io/group.order"                      = "100"
      "alb.ingress.kubernetes.io/healthcheck-protocol"             = "HTTP"
      "alb.ingress.kubernetes.io/healthcheck-port"                 = "traffic-port"
      "alb.ingress.kubernetes.io/healthcheck-interval-seconds"     = "15"
      "alb.ingress.kubernetes.io/healthcheck-timeout-seconds"      = "5"
      "alb.ingress.kubernetes.io/success-codes"                    = "200"
      "alb.ingress.kubernetes.io/healthy-threshold-count"          = "2"
      "alb.ingress.kubernetes.io/unhealthy-threshold-count"        = "2"
      # SSL Configuration
      "alb.ingress.kubernetes.io/listen-ports"    = "[{\"HTTP\": 80}, {\"HTTPS\":443}]"
      "alb.ingress.kubernetes.io/certificate-arn" = var.certificate_arn
      "alb.ingress.kubernetes.io/ssl-policy"      = "ELBSecurityPolicy-TLS13-1-2-Res-2021-06"
      "alb.ingress.kubernetes.io/actions.multi-forward" = jsonencode({
        Type = "forward",
        ForwardConfig = {
          TargetGroups = [
            for svc in local.service_map : {
              ServiceName = "${svc.lower_name}-mcp-service"
              ServicePort = 8000
              Weight      = 1
            }
          ]
          TargetGroupStickinessConfig = {
            Enabled = false
          }
        }
      })
    }
  }

  spec {
    rule {
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "multi-forward"
              port {
                name = "use-annotation"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_service.mcp_search_services
  ]
}
