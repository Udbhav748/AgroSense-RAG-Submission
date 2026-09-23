import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import VoiceInteractionBar, {
  SpeechToTextButton,
  FieldProtocolAudioPlayer,
  LanguageSelectorDropdown,
  buildEmergencyFieldProtocolText,
} from './VoiceInteractionBar'
import { SUPPORTED_LANGUAGES, getVoiceLanguage } from '../../utils/i18n'

describe('VoiceInteractionBar & Multilingual Accessibility Suite', () => {
  const mockDiagnosis = {
    crop: 'tomato',
    disease: 'early blight',
    confidence: 0.94,
    raw_class: 'Tomato___Early_blight',
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('buildEmergencyFieldProtocolText', () => {
    it('generates a detailed emergency 24-48h script for diseased crop in English', () => {
      const script = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'en')
      expect(script).toContain('Emergency 24 to 48-Hour Field Protocol for tomato affected by early blight')
      expect(script).toContain('Severity level: Severe')
      expect(script).toContain('94%')
      expect(script).toContain('Immediate Step 1: Cultural Sanitation')
      expect(script).toContain('Immediate Step 2: Biological Control')
      expect(script).toContain('Immediate Step 3: Chemical Treatment')
      expect(script).toContain('Safety Advisory: Always wear personal protective equipment')
    })

    it('generates healthy status reassurance script when disease is healthy', () => {
      const script = buildEmergencyFieldProtocolText({
        crop: 'apple',
        disease: 'healthy',
        confidence: 0.99,
      })
      expect(script).toContain('Field Health Report: Your apple foliage is diagnosed as healthy')
      expect(script).toContain('No chemical fungicide is required')
    })

    it('generates localized emergency scripts for Spanish, Hindi, Portuguese, French, and Swahili', () => {
      // Spanish
      const esScript = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'es')
      expect(esScript).toContain('Protocolo de Campo de Emergencia')
      expect(esScript).toContain('Paso Inmediato 1: Saneamiento Cultural')
      expect(esScript).toContain('Aviso de Seguridad')

      // Hindi
      const hiScript = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'hi')
      expect(hiScript).toContain('आपातकालीन 24 से 48 घंटे का फील्ड प्रोटोकॉल')
      expect(hiScript).toContain('तत्काल चरण 1: सांस्कृतिक स्वच्छता')
      expect(hiScript).toContain('सुरक्षा सलाह')

      // Portuguese
      const ptScript = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'pt')
      expect(ptScript).toContain('Protocolo de Emergência de 24 a 48 Horas')
      expect(ptScript).toContain('Passo Imediato 1: Sanitização Cultural')
      expect(ptScript).toContain('Aviso de Segurança')

      // French
      const frScript = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'fr')
      expect(frScript).toContain("Protocole de Terrain d'Urgence")
      expect(frScript).toContain('Étape Immédiate 1 : Mesures Prophylactiques')
      expect(frScript).toContain('Avis de Sécurité')

      // Swahili
      const swScript = buildEmergencyFieldProtocolText(mockDiagnosis, null, 'sw')
      expect(swScript).toContain('Itifaki ya Dharura ya Saa 24 hadi 48')
      expect(swScript).toContain('Hatua ya Haraka 1: Usafi wa Shamba')
      expect(swScript).toContain('Ushauri wa Usalama')
    })

    it('handles empty or missing diagnosis safely', () => {
      const script = buildEmergencyFieldProtocolText(null)
      expect(script).toBe('No active crop diagnosis available.')
    })
  })

  describe('SpeechToTextButton (STT)', () => {
    it('renders with accessible aria labels and toggles listening state on click', async () => {
      const handleTranscript = vi.fn()
      render(<SpeechToTextButton onTranscript={handleTranscript} language="en" />)

      const micButton = screen.getByTestId('speech-to-text-mic-button')
      expect(micButton).toBeInTheDocument()
      expect(micButton).toHaveAttribute('aria-label', 'Start hands-free voice dictation')

      fireEvent.click(micButton)

      await waitFor(() => {
        expect(screen.getByText(/Listening\.\.\. Speak clearly/i)).toBeInTheDocument()
      })
      expect(micButton).toHaveAttribute('aria-label', 'Stop voice recording')

      // Clicking again stops listening
      fireEvent.click(micButton)
      await waitFor(() => {
        expect(screen.queryByText(/Listening\.\.\. Speak clearly/i)).not.toBeInTheDocument()
      })
    })

    it('configures recognition language code for each supported language', () => {
      const languages = [
        { code: 'en', voice: 'en-US' },
        { code: 'es', voice: 'es-ES' },
        { code: 'hi', voice: 'hi-IN' },
        { code: 'pt', voice: 'pt-BR' },
        { code: 'fr', voice: 'fr-FR' },
        { code: 'sw', voice: 'sw-KE' },
      ]

      languages.forEach(({ code, voice }) => {
        expect(getVoiceLanguage(code)).toBe(voice)
      })
    })

    it('gracefully handles unsupported browsers', () => {
      const originalSR = window.SpeechRecognition
      const originalWSR = window.webkitSpeechRecognition
      delete window.SpeechRecognition
      delete window.webkitSpeechRecognition

      render(<SpeechToTextButton onTranscript={vi.fn()} />)

      const micButton = screen.getByTestId('speech-to-text-mic-button')
      expect(micButton).toBeDisabled()
      expect(micButton).toHaveAttribute('aria-label', 'Voice input not supported in this browser')

      // Restore
      window.SpeechRecognition = originalSR
      window.webkitSpeechRecognition = originalWSR
    })
  })

  describe('FieldProtocolAudioPlayer (TTS)', () => {
    it('renders soundwaves and triggers speech narration with selected language voice code', async () => {
      render(<FieldProtocolAudioPlayer diagnosis={mockDiagnosis} language="es" />)

      expect(screen.getByTestId('field-protocol-audio-player')).toBeInTheDocument()
      expect(screen.getByTestId('audio-soundwave-bars')).toBeInTheDocument()

      const playBtn = screen.getByTestId('voice-play-protocol-button')
      expect(playBtn).toBeInTheDocument()

      // 1. Play in Spanish
      fireEvent.click(playBtn)
      expect(window.speechSynthesis.speak).toHaveBeenCalledWith(
        expect.objectContaining({
          text: expect.stringContaining('Protocolo de Campo de Emergencia'),
          lang: 'es-ES',
          rate: 0.95,
        })
      )

      // 2. Pause
      await waitFor(() => {
        expect(screen.getByTestId('voice-pause-protocol-button')).toBeInTheDocument()
      })
      const pauseBtn = screen.getByTestId('voice-pause-protocol-button')
      fireEvent.click(pauseBtn)
      expect(window.speechSynthesis.pause).toHaveBeenCalled()

      // 3. Resume
      const resumeBtn = screen.getByTestId('voice-play-protocol-button')
      expect(resumeBtn).toHaveTextContent('Reanudar')
      fireEvent.click(resumeBtn)
      expect(window.speechSynthesis.resume).toHaveBeenCalled()

      // 4. Stop
      const stopBtn = screen.getByTestId('voice-stop-protocol-button')
      fireEvent.click(stopBtn)
      expect(window.speechSynthesis.cancel).toHaveBeenCalled()
    })

    it('renders unsupported banner if speechSynthesis is unavailable', () => {
      const originalSpeechSynthesis = window.speechSynthesis
      delete window.speechSynthesis

      render(<FieldProtocolAudioPlayer diagnosis={mockDiagnosis} />)

      expect(screen.getByText('Audio narration unsupported in this browser')).toBeInTheDocument()

      // Restore
      window.speechSynthesis = originalSpeechSynthesis
    })
  })

  describe('LanguageSelectorDropdown', () => {
    it('renders 6 global agricultural languages and notifies parent on selection change', () => {
      const handleChange = vi.fn()
      render(<LanguageSelectorDropdown value="en" onChange={handleChange} />)

      const select = screen.getByTestId('voice-language-selector')
      expect(select).toBeInTheDocument()
      expect(select).toHaveValue('en')

      // Check all 6 languages are options
      expect(screen.getByRole('option', { name: /English/i })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: /Español/i })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: /हिन्दी/i })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: /Português/i })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: /Français/i })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: /Kiswahili/i })).toBeInTheDocument()

      // Switch language to Hindi
      fireEvent.change(select, { target: { value: 'hi' } })
      expect(handleChange).toHaveBeenCalledWith('hi')
    })
  })

  describe('Composite VoiceInteractionBar', () => {
    it('renders language selector, audio player, and dictation section with language change propagation', () => {
      const handleTranscript = vi.fn()
      const handleLanguageChange = vi.fn()
      render(
        <VoiceInteractionBar
          diagnosis={mockDiagnosis}
          onTranscript={handleTranscript}
          onLanguageChange={handleLanguageChange}
          language="en"
        />
      )

      expect(screen.getByTestId('voice-interaction-bar')).toBeInTheDocument()
      expect(screen.getByTestId('voice-language-selector')).toBeInTheDocument()
      expect(screen.getByTestId('field-protocol-audio-player')).toBeInTheDocument()
      expect(screen.getByText(/Hands-free Field Voice Dictation/i)).toBeInTheDocument()
      expect(screen.getByTestId('speech-to-text-mic-button')).toBeInTheDocument()

      // Change language via selector dropdown
      const langSelect = screen.getByTestId('voice-language-selector')
      fireEvent.change(langSelect, { target: { value: 'sw' } })
      expect(handleLanguageChange).toHaveBeenCalledWith('sw')
    })
  })
})
