https://libertymutual.atlassian.net/wiki/spaces/GSAG/pages/2602893559/VAIPR+2.0+Exposed+Limit+Calc

Exposed Limit Calculation – Overview
The Exposed Limit calculation determines how much of the final contract/layer-adjusted loss* should be attributed back to each individual location.

In simple terms:

We start with location-level gross loss, apply sublimit terms and layer terms at grouped levels, calculate a final contract-level loss, and then allocate that loss back to locations proportionally.

 

*loss: In this context, loss means maximum loss potential, not claims info

 

Example 
This example shows how exposed limit is calculated for a selected set of locations within one contract and one peril.

Assumptions:

Selected locations are L1, L2, L3, and L4.

All values are already in USD.

No facultative reinsurance is applied in this example.

No location-level limit is binding in this example.

The policy deductible is used as a cap on summed location deductibles within a group.

In this example, location_gross_loss_amount is approximated as TIV minus location deductible.

 

Definitions:

TIV_i = total insured value for location i

LD_i = location deductible for location i

GL_i = location_gross_loss_amount for location i

Sum_Loc_GRLoss = total location gross loss across the selected locations

SL_g = sublimit for group g

PD = policy deductible

PA = policy attachment point

PL = policy limit

PP = policy participation

Contract_GR_Loss = final contract gross loss after sublimit and layer terms

GR_ExpLim_i = exposed limit allocated to location i


 

Input Data


Where:

(SL_g) = sublimit for group (g)

(PA) = layer attachment point

(PL) = layer occurrence limit

(PP) = layer participation

Step 1: Calculate location gross loss amount


 

 

Step 2: Apply Sublimit Terms


 

Step 3: Apply Layer Terms


 

Step 4: Back-Allocate Final Contract Gross Loss to Selected Locations

To match the selected-location output, allocate the final contract gross loss back to each location by location gross loss share:


 

Output


 

The example calculates the contract-level exposed limit by grouping locations by sublimit, applying policy attachment, policy limit, and participation, and producing a final contract gross loss of 80. The selected-location output then back-allocates that Contract_GR_Loss to each location in proportion to the location's share of total location gross loss.

 

Inputs:

For each selected location:

Replacement values (RV1, RV2, RV3, RV4)

Total replacement value (TRV)

Location deductible type and deductible amounts

Location limit type and limit amounts

Location participation values

Sublimit inputs:

Sublimit amount

Sublimit attachment

Sublimit minimum and maximum deductible rules

Layer inputs:

Layer attachment point

Layer occurrence limit

Layer participation

Layer deductible rules

…

 

Output

contract_number

insured_name

address_text

city

area_code

postal_code

country_2_digit_iso_code

country_name

cede_db

peril

original_currency_code

exchange_rate_used

TIV

location_gross_loss_amount

Sum_Loc_GRLoss

location_deductible_amount

location_interim_limit_amount

sublimit_attachment_point

layer_occupation_total_limit_amount

layer_occupation_participation

layer_attachment_point

Contract_GR_Loss

GR_ExpLim

 

High-Level Flow
The calculation happens in 4 main stages:

Filter the data to the relevant peril

Apply sublimit terms

Apply layer terms

Allocate the final adjusted loss back to each location

Step 1: Filter to the relevant exposure records
The process begins by selecting exposure records for the required peril:

E.g., records where peril = 'FF' 

Data comes from:

vaipr_location

vaipr_location_exposure

This creates the base dataset used throughout the rest of the calculation.

Step 2: Apply Sublimit Terms
The code then groups losses at the sublimit correlation/group level using:

contract_key

layer_sid

limit_correlation_id

cede_db

At this grouped level, the following are calculated:

Total location deductible

Total gross loss

Sublimit amount

Minimum/maximum deductible constraints

Sublimit attachment point

What happens in this stage?
2.1 Determine the sublimit deductible
The deductible used for the group is adjusted based on:

minimum deductible

maximum deductible

whether the deductible is stored as:

a flat amount, or

a percentage of gross loss

2.2 Apply the sublimit attachment
If an attachment exists:

Loss below the attachment is reduced to zero

Loss above the attachment is reduced by the attachment amount

2.3 Adjust for deductible differences
Because some deductible has already been applied at location level, the code adjusts the grouped loss so that the final deductible at sublimit level is consistent.

2.4 Apply the sublimit cap
If the adjusted grouped loss exceeds the sublimit, it is capped at the sublimit amount.

Business meaning
At the end of this stage, each sublimit group has a post-sublimit adjusted loss.

Step 3: Apply Layer Terms
After sublimit processing, the code rolls the results up to the layer level.

This uses:

contract_key

layer_sid

cede_db

It also joins in the relevant layer configuration, including:

layer deductible type

attachment point

occurrence limit

participation

deductible values

What happens in this stage?
3.1 Calculate the layer deductible
The deductible applied depends on layer_deductible_type_code.

The code supports different deductible styles, such as:

flat deductible

minimum deductible

maximum deductible

percentage-of-loss deductible

franchise/attachment-based handling

3.2 Adjust the loss for the layer deductible
If the layer deductible differs from what was already applied during sublimit processing, the grouped loss is adjusted accordingly.

3.3 Apply the layer attachment point
If a layer attachment exists:

losses below the attachment are removed

losses above the attachment are reduced by the attachment amount

3.4 Apply occurrence limit and participation
The adjusted layer loss is then subject to:

occurrence total limit

participation percentage

This means:

if the loss exceeds the occurrence limit, it is capped

then only the participating share is retained

Business meaning
At the end of this stage, the model has a final layer-adjusted loss.

Step 4: Roll up to Contract Level
Once all layer terms have been applied, the code sums the final layer-adjusted losses by contract_key.

Business meaning
This is the total loss remaining for the contract after all sublimit and layer rules have been applied.

Step 5: Allocate the Final Contract Loss Back to Locations
This is the final step where Exposed Limit is created.

The code:

Calculates the total original location gross loss for the contract.

Allocates the final contract loss back to each location based on that location’s share of the original gross loss

Final allocation formula
