import '@testing-library/jest-dom'
import { randomUUID } from 'node:crypto'

// jsdom's built-in Crypto implementation doesn't implement randomUUID();
// useChat.js relies on it to mint session ids, so patch it in for tests.
if (typeof globalThis.crypto?.randomUUID !== 'function') {
  globalThis.crypto = { ...(globalThis.crypto ?? {}), randomUUID }
}

if (typeof globalThis.DOMMatrix === 'undefined') {
  globalThis.DOMMatrix = class DOMMatrix {}
}

if (typeof window !== 'undefined') {
  window.DOMMatrix = globalThis.DOMMatrix
  Element.prototype.scrollIntoView = Element.prototype.scrollIntoView || vi.fn()

  const mockCreateObjectURL = vi.fn(() => 'blob:http://localhost/mock-leaf-preview')
  const mockRevokeObjectURL = vi.fn()
  window.URL.createObjectURL = mockCreateObjectURL
  window.URL.revokeObjectURL = mockRevokeObjectURL
  globalThis.URL.createObjectURL = mockCreateObjectURL
  globalThis.URL.revokeObjectURL = mockRevokeObjectURL

  // --- Web Speech API (STT: SpeechRecognition) Mock ---
  class MockSpeechRecognition {
    constructor() {
      this.continuous = false
      this.interimResults = true
      this.lang = 'en-US'
      this.onstart = null
      this.onresult = null
      this.onerror = null
      this.onend = null
    }

    start() {
      this.onstart?.()
    }

    stop() {
      this.onend?.()
    }

    abort() {
      this.onend?.()
    }
  }

  window.SpeechRecognition = MockSpeechRecognition
  window.webkitSpeechRecognition = MockSpeechRecognition
  globalThis.SpeechRecognition = MockSpeechRecognition
  globalThis.webkitSpeechRecognition = MockSpeechRecognition

  // --- SpeechSynthesis API (TTS) Mock ---
  class MockSpeechSynthesisUtterance {
    constructor(text = '') {
      this.text = text
      this.lang = 'en-US'
      this.pitch = 1
      this.rate = 1
      this.volume = 1
      this.voice = null
      this.onstart = null
      this.onend = null
      this.onerror = null
      this.onpause = null
      this.onresume = null
    }
  }

  const mockSpeechSynthesis = {
    speak: vi.fn((utterance) => {
      mockSpeechSynthesis.speaking = true
      mockSpeechSynthesis.paused = false
      utterance?.onstart?.()
    }),
    cancel: vi.fn(() => {
      mockSpeechSynthesis.speaking = false
      mockSpeechSynthesis.paused = false
    }),
    pause: vi.fn(() => {
      mockSpeechSynthesis.paused = true
    }),
    resume: vi.fn(() => {
      mockSpeechSynthesis.paused = false
    }),
    getVoices: vi.fn(() => []),
    speaking: false,
    paused: false,
    pending: false,
    onvoiceschanged: null,
  }

  window.SpeechSynthesisUtterance = MockSpeechSynthesisUtterance
  globalThis.SpeechSynthesisUtterance = MockSpeechSynthesisUtterance
  window.speechSynthesis = mockSpeechSynthesis
  globalThis.speechSynthesis = mockSpeechSynthesis
}

