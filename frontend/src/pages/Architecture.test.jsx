import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Architecture from './Architecture'

describe('Architecture Page - System Architecture & Production AI Design Review', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders header, title, and initial System Architecture Blueprints tab by default', () => {
    render(<Architecture />)

    expect(screen.getByText('System Architecture & Production AI Design Review')).toBeInTheDocument()
    expect(screen.getByText('System Architecture Blueprints')).toBeInTheDocument()
    expect(screen.getByText('10 Production AI Questions')).toBeInTheDocument()
    expect(screen.getByText('LeafSense Vision & Confusion Matrix')).toBeInTheDocument()
    expect(screen.getByText('Vector Space & Knowledge Graph')).toBeInTheDocument()
    expect(screen.getByText('Multimodal RAG & StateGraph')).toBeInTheDocument()
    expect(screen.getByText('Security, Cost & Scaling')).toBeInTheDocument()

    // Architecture topology diagram is present by default
    expect(screen.getByText('Interactive System Architecture Topology')).toBeInTheDocument()
    expect(screen.getByText('Frontend Client & Native Mobile PWA')).toBeInTheDocument()
    expect(screen.getByText('FastAPI Async Gateway & Security Core')).toBeInTheDocument()
    expect(screen.getByText('Hybrid Fusion (RRF)')).toBeInTheDocument()
  })

  it('allows clicking different architecture nodes to inspect technical specifications', () => {
    render(<Architecture />)

    // Click LeafSense Vision node
    const visionNode = screen.getByText('LeafSense Vision')
    fireEvent.click(visionNode)

    expect(screen.getAllByText('LeafSense Deep Vision Microservice').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/38-class agricultural pathology neural network/)).toBeInTheDocument()
  })

  it('switches to 10 Production AI Questions tab and expands questions', () => {
    render(<Architecture />)

    const reviewTabBtn = screen.getByText('10 Production AI Questions')
    fireEvent.click(reviewTabBtn)

    expect(screen.getByText('Production AI Design Review (Ten Framework Questions)')).toBeInTheDocument()

    // Expand Question 2
    const q2Header = screen.getByText('What decisions are delegated to the LLM?')
    fireEvent.click(q2Header)

    expect(screen.getByText('Delegated to LLM')).toBeInTheDocument()
    expect(screen.getByText('Strictly Deterministic (Code / DB)')).toBeInTheDocument()
  })

  it('switches to LeafSense Vision & Confusion Matrix tab and changes crop views', () => {
    render(<Architecture />)

    const visionTabBtn = screen.getByText('LeafSense Vision & Confusion Matrix')
    fireEvent.click(visionTabBtn)

    expect(screen.getByText('38 Classes')).toBeInTheDocument()
    expect(screen.getByText('98.24%')).toBeInTheDocument()
    expect(screen.getByText('Interactive Confusion Matrix & Per-Class Precision')).toBeInTheDocument()

    // Switch crop to Apple
    const appleBtn = screen.getByRole('button', { name: 'Apple' })
    fireEvent.click(appleBtn)

    expect(screen.getByText(/Apple Model Accuracy:/)).toBeInTheDocument()
    expect(screen.getAllByText('Apple Scab').length).toBeGreaterThanOrEqual(1)
  })

  it('switches to Vector Space & Knowledge Graph tab and tests cosine probe', () => {
    render(<Architecture />)

    const vectorTabBtn = screen.getByText('Vector Space & Knowledge Graph')
    fireEvent.click(vectorTabBtn)

    expect(screen.getByText('Vector Embedding Space & Semantic Knowledge Graph')).toBeInTheDocument()
    expect(screen.getByText(/Inner Product Normalization in FAISS IndexFlatIP/)).toBeInTheDocument()
    expect(screen.getByText('Live Query Vector Probe (Cosine Distance Simulator)')).toBeInTheDocument()
    expect(screen.getByText('Top Retrieval Candidates')).toBeInTheDocument()

    // Probe bacterial spot query
    const probeBtn = screen.getByText('Water-soaked angular bacterial spots with copper dosage')
    fireEvent.click(probeBtn)

    expect(screen.getAllByText('Tomato Bacterial Spot').length).toBeGreaterThanOrEqual(1)
  })

  it('switches to Multimodal RAG & StateGraph tab and displays RRF formulas', () => {
    render(<Architecture />)

    const ragTabBtn = screen.getByText('Multimodal RAG & StateGraph')
    fireEvent.click(ragTabBtn)

    expect(screen.getByText('Multi-Agent StateGraph & Corrective RAG Runtime')).toBeInTheDocument()
    expect(screen.getByText('01. Vision Inference')).toBeInTheDocument()
    expect(screen.getByText('03. Hybrid Search (RRF)')).toBeInTheDocument()
    expect(screen.getByText('Reciprocal Rank Fusion (RRF) Formulation')).toBeInTheDocument()
  })

  it('switches to Security, Cost & Scaling tab', () => {
    render(<Architecture />)

    const secTabBtn = screen.getByText('Security, Cost & Scaling')
    fireEvent.click(secTabBtn)

    expect(screen.getByText('Multi-Tenant Isolation Architecture')).toBeInTheDocument()
    expect(screen.getByText('10 to 1,000,000 Scaling Roadmap')).toBeInTheDocument()
  })
})
