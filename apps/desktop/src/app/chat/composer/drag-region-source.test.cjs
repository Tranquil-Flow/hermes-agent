const assert = require('node:assert/strict')
const { readFileSync } = require('node:fs')
const { join } = require('node:path')
const { describe, it } = require('node:test')

const source = readFileSync(join(__dirname, 'index.tsx'), 'utf8')

describe('composer popout drag region layering', () => {
  it('raises only narrow drag hit targets above the composer surface', () => {
    assert.match(source, /data-slot="composer-drag-region"/)
    assert.match(source, /pointer-events-none absolute inset-0 z-5/)
    assert.match(source, /data-slot="composer-drag-hit-target-top"/)
    assert.match(source, /pointer-events-auto absolute inset-x-0 top-0 h-3/)
  })

  it('does not blanket-capture input and control clicks over the whole surface', () => {
    const dragRegionStart = source.indexOf('data-slot="composer-drag-region"')
    const dragRegionEnd = source.indexOf('data-slot="composer-surface"')
    const dragRegionBlock = source.slice(dragRegionStart - 250, dragRegionEnd)

    assert.doesNotMatch(dragRegionBlock, /pointer-events-auto absolute inset-0/)
    assert.match(dragRegionBlock, /pointer-events-none absolute inset-0 z-5/)
  })

  it('keeps dragging cursor state on the raised hit target', () => {
    assert.match(source, /dragging \? 'cursor-grabbing' : 'cursor-grab'/)
  })
})
