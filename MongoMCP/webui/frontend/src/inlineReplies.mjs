export function remarkQuickReplies({ replies = [] } = {}) {
  return tree => {
    const used = new Set()
    const byId = new Map(replies.flatMap((reply, index) => reply.id ? [[reply.id, index]] : []))
    const linkFor = index => ({
      type: 'link',
      url: `#quick-reply-${index}`,
      data: { hProperties: { 'data-quick-reply-index': index } },
      children: [{ type: 'text', value: replies[index].label }],
    })

    function visit(node) {
      if (['code', 'inlineCode', 'link', 'linkReference', 'html'].includes(node.type)) return
      if (!Array.isArray(node.children)) return
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text') {
          visit(child)
          return [child]
        }
        const parts = []
        let offset = 0
        for (const match of child.value.matchAll(/\{\{reply:([A-Za-z0-9_-]{1,64})\}\}/g)) {
          const index = byId.get(match[1])
          if (index === undefined) continue
          if (match.index > offset) parts.push({ type: 'text', value: child.value.slice(offset, match.index) })
          parts.push(linkFor(index))
          used.add(index)
          offset = match.index + match[0].length
        }
        if (offset === 0) return [child]
        if (offset < child.value.length) parts.push({ type: 'text', value: child.value.slice(offset) })
        return parts
      })
    }

    visit(tree)
    const unplaced = replies.flatMap((reply, index) => used.has(index) ? [] : [linkFor(index)])
    if (unplaced.length) {
      tree.children.push({
        type: 'paragraph',
        data: { hProperties: { className: ['quick-replies'], role: 'group', 'aria-label': 'Suggested replies' } },
        children: unplaced,
      })
    }
  }
}