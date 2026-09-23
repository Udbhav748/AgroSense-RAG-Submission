import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import AgentGraphVisualizer, { GRAPH_NODES, StateDrawer } from './AgentGraphVisualizer'

describe('AgentGraphVisualizer Component', () => {
  it('renders all 4 default workflow nodes in idle state', () => {
    render(<AgentGraphVisualizer />)

    expect(screen.getByTestId('agent-graph-visualizer')).toBeInTheDocument()
    expect(screen.getByText(/Multi-Agent Execution Graph/i)).toBeInTheDocument()

    GRAPH_NODES.forEach((node) => {
      expect(screen.getByTestId(`graph-node-${node.id}`)).toBeInTheDocument()
      expect(screen.getByText(node.displayName)).toBeInTheDocument()
    })

    const idleBadges = screen.getAllByTestId('status-badge-idle')
    expect(idleBadges.length).toBe(4)
  })

  it('renders running and completed node states with durations and pipeline active badge', () => {
    const nodeStates = {
      planner: { status: 'completed', duration_ms: 45, output: { action: 'retrieve' } },
      document_analyst: { status: 'running' },
      fact_checker: { status: 'idle' },
      synthesizer: { status: 'idle' },
    }

    render(
      <AgentGraphVisualizer
        nodeStates={nodeStates}
        activeNode="document_analyst"
        isExecuting={true}
      />
    )

    expect(screen.getByTestId('pipeline-running-badge')).toBeInTheDocument()
    expect(screen.getByTestId('status-badge-completed')).toBeInTheDocument()
    expect(screen.getByTestId('status-badge-running')).toBeInTheDocument()
    expect(screen.getByTestId('duration-badge-planner')).toHaveTextContent('45ms')
  })

  it('opens state drawer on node click and renders intermediate state details', () => {
    const nodeStates = {
      fact_checker: {
        status: 'completed',
        duration_ms: 82,
        output: {
          grounded: true,
          score: 0.96,
        },
      },
    }

    render(<AgentGraphVisualizer nodeStates={nodeStates} />)

    const factCheckerNode = screen.getByTestId('graph-node-fact_checker')
    fireEvent.click(factCheckerNode)

    expect(screen.getByTestId('state-drawer')).toBeInTheDocument()
    expect(screen.getByText('Fact Checker State')).toBeInTheDocument()
    expect(screen.getByTestId('state-drawer-duration')).toHaveTextContent('82ms')
    expect(screen.getByTestId('fact-check-grounded')).toHaveTextContent('Verified Grounded')
    expect(screen.getByTestId('fact-check-score')).toHaveTextContent('96%')
  })

  it('renders StateDrawer standalone with custom node payload', () => {
    const node = GRAPH_NODES.find((n) => n.id === 'fact_checker')
    const stateData = {
      status: 'completed',
      duration_ms: 82,
      output: { grounded: true, score: 0.96 },
    }
    const onClose = vi.fn()

    render(<StateDrawer node={node} stateData={stateData} onClose={onClose} />)

    expect(screen.getByTestId('state-drawer')).toBeInTheDocument()
    const closeBtn = screen.getByTestId('state-drawer-close')
    fireEvent.click(closeBtn)
    expect(onClose).toHaveBeenCalled()
  })
})
