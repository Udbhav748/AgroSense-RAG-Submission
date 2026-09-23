import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import Documents from './Documents'
import * as documentService from '../services/documentService'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}))

const mockShowToast = vi.fn()
vi.mock('../hooks/useToast', () => ({
  default: () => ({ showToast: mockShowToast }),
}))

let mockUser = { id: '1', role: 'user' }
vi.mock('../hooks/useAuth', () => ({
  default: () => ({ user: mockUser }),
}))

vi.mock('../services/documentService', () => ({
  listDocuments: vi.fn(),
  getUploadHistory: vi.fn(),
  deleteDocument: vi.fn(),
  removeFromUploadHistory: vi.fn(),
}))

describe('Documents Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUser = { id: '1', role: 'user' }
    documentService.getUploadHistory.mockReturnValue([])
  })

  it('renders loading skeletons initially', () => {
    documentService.listDocuments.mockReturnValue(new Promise(() => {}))

    render(<Documents />)

    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('Loading…')).toBeInTheDocument()
  })

  it('renders empty state when no documents are returned', async () => {
    documentService.listDocuments.mockResolvedValueOnce([])

    render(<Documents />)

    await waitFor(() => {
      expect(screen.getByText('No documents yet')).toBeInTheDocument()
      expect(
        screen.getByText('Upload your first PDF to start building your knowledge base.')
      ).toBeInTheDocument()
    })

    const uploadBtn = screen.getByRole('button', { name: 'Upload a document' })
    fireEvent.click(uploadBtn)
    expect(mockNavigate).toHaveBeenCalledWith('/upload')
  })

  it('renders document list with metadata and count', async () => {
    const mockServerDocs = [
      {
        document_id: 'doc-1',
        original_filename: 'Annual_Report_2025.pdf',
        total_pages: 12,
        total_chunks: 48,
        upload_timestamp: '2026-01-15T10:00:00Z',
        collection: 'Finance',
        tenant_id: 1,
        status: 'processed',
        images_captioned: 2,
        total_tables: 3,
      },
      {
        document_id: 'doc-2',
        original_filename: 'User_Manual.pdf',
        total_pages: 5,
        total_chunks: 15,
        upload_timestamp: '2026-02-01T12:00:00Z',
        collection: 'Engineering',
        tenant_id: 1,
        status: 'processed',
        images_captioned: 0,
        total_tables: 0,
      },
    ]
    documentService.listDocuments.mockResolvedValueOnce(mockServerDocs)

    render(<Documents />)

    await waitFor(() => {
      expect(screen.getByText('2 documents')).toBeInTheDocument()
      expect(screen.getByText('Annual_Report_2025.pdf')).toBeInTheDocument()
      expect(screen.getByText('User_Manual.pdf')).toBeInTheDocument()
      expect(screen.getAllByText('Finance').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Engineering').length).toBeGreaterThan(0)
      expect(screen.getByText('12 pg · 48 chunks')).toBeInTheDocument()
    })
  })

  it('filters documents by search query and shows empty search state', async () => {
    documentService.listDocuments.mockResolvedValueOnce([
      {
        document_id: 'doc-1',
        original_filename: 'Security_Policy.pdf',
        upload_timestamp: '2026-01-01T00:00:00Z',
      },
      {
        document_id: 'doc-2',
        original_filename: 'Employee_Handbook.pdf',
        upload_timestamp: '2026-01-02T00:00:00Z',
      },
    ])

    render(<Documents />)

    await waitFor(() => {
      expect(screen.getByText('Security_Policy.pdf')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText('Search documents...')
    fireEvent.change(searchInput, { target: { value: 'Security' } })

    expect(screen.getByText('Security_Policy.pdf')).toBeInTheDocument()
    expect(screen.queryByText('Employee_Handbook.pdf')).not.toBeInTheDocument()

    fireEvent.change(searchInput, { target: { value: 'nomatchterm' } })
    expect(screen.getByText('No matches')).toBeInTheDocument()
  })

  it('filters by collection and allows navigating to chat for collection', async () => {
    documentService.listDocuments.mockResolvedValueOnce([
      {
        document_id: 'doc-1',
        original_filename: 'DocA.pdf',
        collection: 'Legal',
        upload_timestamp: '2026-01-01T00:00:00Z',
      },
      {
        document_id: 'doc-2',
        original_filename: 'DocB.pdf',
        collection: 'Finance',
        upload_timestamp: '2026-01-02T00:00:00Z',
      },
    ])

    render(<Documents />)

    await waitFor(() => {
      expect(screen.getByText('DocA.pdf')).toBeInTheDocument()
    })

    const collectionSelect = screen.getByRole('combobox', { name: 'Filter by collection' })
    fireEvent.change(collectionSelect, { target: { value: 'Legal' } })

    expect(screen.getByText('DocA.pdf')).toBeInTheDocument()
    expect(screen.queryByText('DocB.pdf')).not.toBeInTheDocument()

    const chatCollectionBtn = screen.getByRole('button', { name: /chat about this collection/i })
    expect(chatCollectionBtn).toBeInTheDocument()
    fireEvent.click(chatCollectionBtn)

    expect(mockNavigate).toHaveBeenCalledWith('/chat', {
      state: { documentIds: ['doc-1'] },
    })
  })

  it('opens confirmation modal on delete and deletes document on confirmation', async () => {
    documentService.listDocuments.mockResolvedValueOnce([
      {
        document_id: 'del-1',
        original_filename: 'Obsolete_Doc.pdf',
        upload_timestamp: '2026-01-01T00:00:00Z',
      },
    ])
    documentService.deleteDocument.mockResolvedValueOnce({ status: 'ok', chunks_removed: 5 })

    render(<Documents />)

    await waitFor(() => {
      expect(screen.getByText('Obsolete_Doc.pdf')).toBeInTheDocument()
    })

    const deleteBtn = screen.getByRole('button', { name: 'Delete Obsolete_Doc.pdf' })
    fireEvent.click(deleteBtn)

    expect(screen.getByRole('dialog', { name: 'Remove document' })).toBeInTheDocument()
    expect(screen.getByText(/permanently removes it from the server/)).toBeInTheDocument()

    const confirmBtn = screen.getByRole('button', { name: 'Delete' })
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(documentService.deleteDocument).toHaveBeenCalledWith('del-1')
      expect(documentService.removeFromUploadHistory).toHaveBeenCalledWith('del-1')
      expect(mockShowToast).toHaveBeenCalledWith('Obsolete_Doc.pdf deleted.', 'success')
      expect(screen.queryByText('Obsolete_Doc.pdf')).not.toBeInTheDocument()
    })
  })

  it('renders all-tenants toggle for admin users', async () => {
    mockUser = { id: 'admin-1', role: 'admin' }
    documentService.listDocuments.mockResolvedValue([])

    render(<Documents />)

    const allTenantsBtn = screen.getByRole('button', { name: /all tenants/i })
    expect(allTenantsBtn).toBeInTheDocument()

    fireEvent.click(allTenantsBtn)
    expect(documentService.listDocuments).toHaveBeenCalledWith({ allTenants: true })
  })
})
