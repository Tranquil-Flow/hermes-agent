const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { parseStageResult, runBootstrap } = require('./bootstrap-runner.cjs')

function makeInstallerScript(body) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-runner-'))
  const scriptsDir = path.join(root, 'scripts')
  fs.mkdirSync(scriptsDir, { recursive: true })
  fs.writeFileSync(path.join(scriptsDir, 'install.sh'), body, { mode: 0o755 })
  return root
}

function makeRunnerDirs() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-home-'))
  return {
    activeRoot: path.join(root, 'agent'),
    hermesHome: path.join(root, 'home'),
    logRoot: path.join(root, 'logs')
  }
}

// ---------------------------------------------------------------------------
// parseStageResult
// ---------------------------------------------------------------------------

test('parseStageResult returns null for empty stdout', () => {
  assert.equal(parseStageResult(''), null)
})

test('parseStageResult extracts last JSON line with ok+stage fields', () => {
  const stdout = 'some noise\n{"ok":true,"stage":"repository"}'
  const result = parseStageResult(stdout)
  assert.deepEqual(result, { ok: true, stage: 'repository' })
})

test('parseStageResult returns null when no JSON line has ok+stage', () => {
  const stdout = '{"foo":1}\n{"bar":2}'
  assert.equal(parseStageResult(stdout), null)
})

test('parseStageResult picks last valid frame when multiple JSON lines present', () => {
  const stdout = '{"ok":false,"stage":"prerequisites"}\n{"ok":true,"stage":"repository"}'
  const result = parseStageResult(stdout)
  assert.deepEqual(result, { ok: true, stage: 'repository' })
})

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []
  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal
  })

  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
})

test('killed bootstrap stage reports captured stderr context', async () => {
  const sourceRepoRoot = makeInstallerScript(`#!/usr/bin/env bash
if [[ "$*" == *"--manifest"* ]]; then
  echo '{"protocol_version":1,"stages":[{"name":"repository","title":"Repository"}]}'
  exit 0
fi
if [[ "$*" == *"--stage repository"* ]]; then
  echo 'Repository ready'
  echo 'fatal: detached HEAD state' >&2
  exec sleep 10
fi
`)
  const controller = new AbortController()
  const events = []
  const resultPromise = runBootstrap({
    installStamp: null,
    sourceRepoRoot,
    ...makeRunnerDirs(),
    onEvent: ev => {
      events.push(ev)
      if (ev.type === 'log' && ev.stage === 'repository' && ev.line.includes('fatal: detached HEAD')) {
        controller.abort()
      }
    },
    abortSignal: controller.signal
  })

  const result = await resultPromise

  assert.equal(result.ok, false)
  assert.equal(result.failedStage, 'repository')
  assert.match(result.error, /process killed — last output:/)
  assert.match(result.error, /fatal: detached HEAD state/)
  assert.doesNotMatch(result.error, /^cancelled by user$/)
})

test('killed bootstrap stage without captured output still reports user cancellation', async () => {
  const sourceRepoRoot = makeInstallerScript(`#!/usr/bin/env bash
if [[ "$*" == *"--manifest"* ]]; then
  echo '{"protocol_version":1,"stages":[{"name":"repository","title":"Repository"}]}'
  exit 0
fi
if [[ "$*" == *"--stage repository"* ]]; then
  exec sleep 10
fi
`)
  const controller = new AbortController()
  const events = []
  const resultPromise = runBootstrap({
    installStamp: null,
    sourceRepoRoot,
    ...makeRunnerDirs(),
    onEvent: ev => {
      events.push(ev)
      if (ev.type === 'stage' && ev.name === 'repository' && ev.state === 'running') {
        controller.abort()
      }
    },
    abortSignal: controller.signal
  })

  const result = await resultPromise

  assert.equal(result.ok, false)
  assert.equal(result.failedStage, 'repository')
  assert.equal(result.error, 'cancelled by user')
})
