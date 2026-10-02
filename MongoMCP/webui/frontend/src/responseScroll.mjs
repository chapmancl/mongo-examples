export function scrollCompletedResponse(container, response, spacer) {
  if (!container || !response || !spacer) return
  const top = Math.max(0, container.scrollTop + response.getBoundingClientRect().top
    - container.getBoundingClientRect().top - container.clientTop)
  const contentHeight = container.scrollHeight - spacer.offsetHeight
  spacer.style.height = `${Math.max(0, top + container.clientHeight - contentHeight)}px`
  container.scrollTo({ top, behavior: 'instant' })
}