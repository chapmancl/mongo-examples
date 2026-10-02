import assert from 'node:assert/strict'
import test from 'node:test'
import { remarkQuickReplies } from './inlineReplies.mjs'

const replies = [
  { id: 'first', label: 'First example', value: 'Run the first prompt' },
  { id: 'second', label: 'Second example', value: 'Run the second prompt' },
]
const text = value => ({ type: 'text', value })
const paragraph = (...children) => ({ type: 'paragraph', children })
const root = (...children) => ({ type: 'root', children })

test('places buttons within nested Markdown and only appends unused choices', () => {
  const tree = root(paragraph(text('Try '), { type: 'strong', children: [text('{{reply:first}}')] }, text(' now.')))
  remarkQuickReplies({ replies })(tree)
  assert.equal(tree.children[0].children[1].children[0].url, '#quick-reply-0')
  assert.equal(tree.children[0].children[1].children[0].data.hProperties['data-quick-reply-index'], 0)
  assert.equal(tree.children[0].children[1].children[0].children[0].value, 'First example')
  assert.equal(tree.children[1].children.length, 1)
  assert.equal(tree.children[1].children[0].url, '#quick-reply-1')
})

test('preserves code, existing links and unknown markers', () => {
  const tree = root(
    { type: 'code', value: '{{reply:first}}' },
    paragraph({ type: 'inlineCode', value: '{{reply:first}}' }),
    paragraph({ type: 'link', url: '#quick-reply-0', children: [text('{{reply:first}}')] }),
    paragraph(text('{{reply:missing}}')),
  )
  const original = structuredClone(tree.children)
  remarkQuickReplies({ replies })(tree)
  assert.deepEqual(tree.children.slice(0, 4), original)
  assert.equal(tree.children[4].children.length, 2)
})

test('multiple markers retain surrounding text without duplicate fallback buttons', () => {
  const tree = root(paragraph(text('A {{reply:first}} B {{reply:second}} C {{reply:first}}.')))
  remarkQuickReplies({ replies })(tree)
  assert.equal(tree.children.length, 1)
  assert.deepEqual(tree.children[0].children.filter(node => node.type === 'text').map(node => node.value),
    ['A ', ' B ', ' C ', '.'])
  assert.equal(tree.children[0].children.filter(node => node.type === 'link').length, 3)
})

test('legacy choices without ids remain available at the end', () => {
  const tree = root(paragraph(text('Welcome!')))
  remarkQuickReplies({ replies: [{ label: 'Legacy option', value: 'Run legacy prompt' }] })(tree)
  assert.equal(tree.children[1].children[0].children[0].value, 'Legacy option')
})

test('no supplied choices means no added controls', () => {
  const tree = root(paragraph(text('Plain answer {{reply:missing}}')))
  const original = structuredClone(tree)
  remarkQuickReplies()(tree)
  assert.deepEqual(tree, original)
})