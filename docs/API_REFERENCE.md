# AgroSense-RAG REST & SSE API Reference

Welcome to the definitive API reference documentation for **AgroSense-RAG**. This reference details all available HTTP REST endpoints and Server-Sent Events (SSE) streaming protocols for agricultural Q&A, plant pathology leaf photo diagnosis, microclimate risk assessment, asynchronous document ingestion, and telemetry.

---

## Table of Contents

1. [Global API Conventions & Authentication](#1-global-api-conventions--authentication)
2. [Error Taxonomy & Standard Error Format](#2-error-taxonomy--standard-error-format)
3. [Interactive & Streaming Chat Endpoints](#3-interactive--streaming-chat-endpoints)
   - [`POST /api/v1/chat/query` (or `/chat`)](#post-apiv1chatquery-or-chat)
   - [`POST /api/v1/chat/stream` (or `/chat/stream`)](#post-apiv1chatstream-or-chatstream)
4. [Plant Pathology Vision Diagnosis Endpoints](#4-plant-pathology-vision-diagnosis-endpoints)
   - [`POST /api/v1/chat/diagnose` (or `/chat/diagnose`)](#post-apiv1chatdiagnose-or-chatdiagnose)
   - [`POST /api/v1/chat/diagnose/stream` (or `/chat/diagnose/stream`)](#post-apiv1chatdiagnosestream-or-chatdiagnosestream)
5. [Agronomic Microclimate & Weather Intelligence](#5-agronomic-microclimate--weather-intelligence)
   - [`GET /api/v1/weather/risk` (or `/weather/risk`)](#get-apiv1weatherrisk-or-weatherrisk)
6. [Asynchronous Document Ingestion & Task Management](#6-asynchronous-document-ingestion--task-management)
   - [`POST /api/v1/upload/async` (or `/upload/async`)](#post-apiv1uploadasync-or-uploadasync)
   - [`GET /api/v1/documents/tasks/{task_id}`](#get-apiv1documentstaskstask_id)
   - [`POST /api/v1/upload` (Synchronous Upload)](#post-apiv1upload-synchronous-upload)
   - [`GET /api/v1/documents`](#get-apiv1documents)
   - [`DELETE /api/v1/documents/{document_id}`](#delete-apiv1documentsdocument_id)
   - [`GET /api/v1/documents/{document_id}/file`](#get-apiv1documentsdocument_idfile)
   - [`GET /api/v1/documents/{document_id}/pages/{page}/highlight`](#get-apiv1documentsdocument_idpagespagehighlight)
7. [Observability & Health Checks](#7-observability--health-checks)
   - [`GET /api/v1/health` (or `/health`)](#get-apiv1health-or-health)
   - [`GET /api/v1/health/vision`](#get-apiv1healthvision)
   - [`GET /api/v1/metrics` (or `/metrics`)](#get-apiv1metrics-or-metrics)
8. [Human-in-the-Loop Feedback](#8-human-in-the-loop-feedback)
   - [`POST /api/v1/feedback` (or `/feedback`)](#post-apiv1feedback-or-feedback)

---

## 1. Global API Conventions & Authentication

### 1.1 Base URLs
- **Local Development**: `http://localhost:8000`
- **Docker Compose Stack**: `http://localhost:80` (via Caddy reverse proxy)
- **Routing Note**: Routes are accessible via direct root paths (e.g. `/chat`, `/health`) as well as versioned API prefixes (`/api/v1/chat/query`, `/api/v1/health`, etc.).

### 1.2 Authentication Schemes
AgroSense-RAG supports two interchangeable authentication methods handled by [`app.core.auth.require_auth`](file:///backend/app/core/auth.py):

1. **JWT Bearer Token (Web SPA)**:
   ```http
   Authorization: Bearer <jwt_token>
   ```
2. **Static API Key (Server / CLI / Automated Pipelines)**:
   ```http
   X-API-Key: <your_api_key>
   ```

### 1.3 Request Tracing & Correlation Headers
- **`X-Request-ID`**: A UUID assigned to every request. If supplied by the client, it is propagated through all log lines and returned in the response header; otherwise, the server generates a fresh UUID.

---

## 2. Error Taxonomy & Standard Error Format

All error responses return a standardized JSON payload:

```json
{
  "detail": "Descriptive human-readable error message.",
  "error_code": "RESOURCE_NOT_FOUND",
  "taxonomy_category": "validation",
  "request_id": "8fa21b4a-d603-4f9e-a612-42e128cb5219"
}
```

### Taxonomy Categories
| Category | HTTP Status | Description |
| :--- | :--- | :--- |
| `auth` | `401 Unauthorized` | Invalid, expired, or missing JWT or API Key. |
| `permission` | `403 Forbidden` | User lacks the required RBAC role (`admin`, `member`, `viewer`). |
| `validation` | `422 Unprocessable Entity` | Pydantic schema validation failure or corrupted payload. |
| `resource` | `404 Not Found` | Document, task, image, or session ID not found. |
| `approval` | `428 Precondition Required` | Action requires explicit human approval flag (`confirm_web_search=true`). |
| `upstream_llm` | `502 Bad Gateway` | Upstream provider timeout, rate limit (429), or authentication error. |
| `internal` | `500 Internal Server Error` | Unhandled backend exception. |

---

## 3. Interactive & Streaming Chat Endpoints

### `POST /api/v1/chat/query` (or `/chat`)

Executes a synchronous natural language question-answering query against the indexed document corpus using the 2-Stage Hybrid RRF + Cross-Encoder retrieval pipeline and Multi-Agent StateGraph.

#### Request Headers
- `Authorization: Bearer <jwt_token>` OR `X-API-Key: <key>`
- `Content-Type: application/json`

#### Request Body Schema (`ChatRequest`)
```json
{
  "query": "What fungicides and active ingredients treat early blight on tomatoes?",
  "session_id": "d1396a84-0a6e-4cb8-b0a3-d023b320875c",
  "top_k": 5,
  "min_score": 0.35,
  "confirm_web_search": false,
  "structured_response": false,
  "persona": "agronomist",
  "document_ids": ["doc-tomato-management-2024"]
}
```

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | — | Natural language question (minimum length: 1). |
| `session_id` | `string` | No | `null` | Multi-turn conversation identifier. Auto-generated if omitted. |
| `top_k` | `integer` | No | `5` | Number of context chunks to retrieve. |
| `min_score` | `float` | No | `0.30` | Minimum similarity score threshold (-1.0 to 1.0). |
| `confirm_web_search` | `boolean`| No | `false` | Human-in-the-loop authorization to invoke external web search tool. |
| `structured_response`| `boolean`| No | `false` | Enforce JSON-mode output if supported by provider. |
| `persona` | `string` | No | `null` | Style preset (`agronomist`, `concise`, `technical`). |
| `document_ids` | `array[str]`| No | `null` | Scope retrieval to specific document IDs. |

#### Response Schema (`ChatResponse` - `200 OK`)
```json
{
  "answer": "Early blight (*Alternaria solani*) on tomatoes is treated with fungicides including Chlorothalonil 75% WP, Azoxystrobin 23% SC, and Difenoconazole 25% EC [1]. Organic alternatives include Copper octanoate and *Bacillus subtilis* [2].",
  "session_id": "d1396a84-0a6e-4cb8-b0a3-d023b320875c",
  "retrieved_chunks": [
    {
      "chunk_id": "doc-tomato-01_chunk_004",
      "document_id": "doc-tomato-01",
      "text": "For tomato early blight (Alternaria solani), apply Chlorothalonil 75% WP at 2.0 kg/ha or Azoxystrobin 23% SC at 500 mL/ha.",
      "score": 0.894,
      "metadata": {"crop": "tomato", "disease": "early blight"}
    }
  ],
  "sources": [
    {
      "number": 1,
      "document_id": "doc-tomato-01",
      "chunk_id": "doc-tomato-01_chunk_004",
      "excerpt": "For tomato early blight (Alternaria solani), apply Chlorothalonil 75% WP at 2.0 kg/ha...",
      "url": null,
      "content_type": "text",
      "page_number": 4
    }
  ],
  "processing_time": 1.342,
  "tool_used": "retrieval",
  "steps_taken": 3,
  "answer_source": "documents",
  "retrieval_confidence": "good",
  "is_clarifying_question": false,
  "follow_up_questions": [
    "What is the pre-harvest interval (PHI) for Azoxystrobin on tomatoes?",
    "How often should Copper octanoate be applied during wet weather?"
  ],
  "hallucination_detected": false,
  "grounding_score": 0.945,
  "weather_risk": null,
  "diagnosis": null
}
```

---

### `POST /api/v1/chat/stream` (or `/chat/stream`)

Streams real-time agent thoughts, retrieval execution traces, and generated answer tokens over Server-Sent Events (SSE).

#### Request Headers
- `Authorization: Bearer <jwt_token>` OR `X-API-Key: <key>`
- `Content-Type: application/json`
- `Accept: text/event-stream`

#### Request Body
Identical to [`ChatRequest`](#request-body-schema-chatrequest).

#### Server-Sent Events Protocol & Wire Format
The server emits `data: <json>\n\n` lines.

```
data: {"type": "trace", "text": "Analyzing query intent and selecting agent strategy..."}

data: {"type": "tool_call", "tool": "retrieval", "payload": {"top_k": 5, "crop": "tomato"}}

data: {"type": "retrieval", "payload": {"chunks_found": 5, "confidence": "good"}}

data: {"type": "answer_chunk", "text": "Early ", "payload": {"token": "Early "}}

data: {"type": "answer_chunk", "text": "blight on ", "payload": {"token": "blight on "}}

data: {"type": "done", "payload": { /* Full ChatResponse Object */ }}
```

#### SSE Event Types Catalog
| Event Type | Description | Payload Structure |
| :--- | :--- | :--- |
| `trace` | Agent StateGraph step transition updates. | `{"text": "Entering planner node..."}` |
| `tool_call` | Notification that an agent node is invoking a tool. | `{"tool": "retrieval", "payload": {...}}` |
| `retrieval` | Retrieval summary and confidence grade. | `{"chunks_found": 5, "confidence": "good"}` |
| `answer_chunk` | Individual token streamed from the LLM. | `{"token": "string"}` |
| `done` | Stream completion event containing the full response. | Complete `ChatResponse` JSON. |
| `error` | Fatal error during pipeline execution. | `{"detail": "error message", "code": "..."}` |

---

## 4. Plant Pathology Vision Diagnosis Endpoints

### `POST /api/v1/chat/diagnose` (or `/chat/diagnose`)

Multipart endpoint allowing farmers to upload a leaf photograph. The image is processed through the cascaded LeafSense CNN / Gemini Vision pipeline, optionally blended with microclimate data, and grounded in the RAG document repository.

#### Request Headers
- `Authorization: Bearer <jwt_token>` OR `X-API-Key: <key>`
- `Content-Type: multipart/form-data`

#### Multipart Form Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | `file` (binary) | **Yes** | Plant leaf image (JPEG, PNG, WebP, max 10MB). |
| `query` | `string` | No | Optional accompanying question (e.g. "Is this organic treatable?"). |
| `session_id` | `string` | No | Multi-turn session ID. |
| `engine` | `string` | No | Vision engine: `hybrid` (default), `leafsense`, or `gemini`. |
| `latitude` | `float` | No | Field latitude (triggers Open-Meteo microclimate evaluation). |
| `longitude` | `float` | No | Field longitude. |
| `confirm_web_search` | `boolean` | No | Authorize web search if local documents lack treatment info. |

#### Response Schema (`ChatResponse` - `200 OK`)
Includes the `diagnosis` object and optional `weather_risk` object:

```json
{
  "answer": "The tomato leaf displays symptoms of **Early Blight** (*Alternaria solani*), diagnosed with 98.4% confidence [1]. Immediate treatment with Chlorothalonil 75% WP or Copper octanoate is recommended.",
  "diagnosis": {
    "raw_class": "Tomato___Early_blight",
    "crop": "tomato",
    "disease": "early blight",
    "confidence": 0.9842,
    "low_confidence": false
  },
  "weather_risk": {
    "location": {"latitude": 43.65, "longitude": -79.38, "timezone": "America/Toronto"},
    "current": {"temperature_c": 19.4, "humidity_pct": 92.0, "precipitation_mm": 0.0, "wind_kmh": 8.2},
    "risk_level": "High",
    "risk_score": 0.88,
    "favorable_conditions_summary": "Smith Period criteria triggered: 11 consecutive hours with RH >= 90% at 19.4°C.",
    "spray_advisory": "Optimal spray window in next 4-6 hours: Calm winds (<15 km/h) and dry conditions forecast."
  },
  "sources": [ ... ],
  "retrieved_chunks": [ ... ],
  "session_id": "f812-421b-...",
  "processing_time": 1.821,
  "tool_used": "diagnose"
}
```

---

### `POST /api/v1/chat/diagnose/stream` (or `/chat/diagnose/stream`)

Streams leaf diagnosis events and token-by-token treatment recommendations over SSE.

#### Server-Sent Events Sequence
1. `{"type": "trace", "text": "Dispatching image to LeafSense Vision CNN..."}`
2. `{"type": "diagnosis", "payload": {"crop": "tomato", "disease": "early blight", "confidence": 0.9842}}`
3. `{"type": "trace", "text": "Retrieving treatment protocols from agronomic corpus..."}`
4. `{"type": "answer_chunk", "text": "Based on the diagnosis..."}`
5. `{"type": "done", "payload": { /* Full ChatResponse */ }}`

---

## 5. Agronomic Microclimate & Weather Intelligence

### `GET /api/v1/weather/risk` (or `/weather/risk`)

Evaluates field microclimate pathogen infection pressure (Smith Periods, Powdery Mildew, Bacterial Spot, Foliar Rust) and generates a chemical spray drift advisory using Open-Meteo forecasts.

#### Query Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `lat` | `float` | **Yes** | Field latitude (e.g. `28.6139`). |
| `lon` | `float` | **Yes** | Field longitude (e.g. `77.2090`). |
| `crop` | `string` | No | Target crop filter (e.g. `tomato`, `potato`, `apple`). |
| `disease` | `string` | No | Target disease filter (e.g. `late blight`). |

#### Response Schema (`WeatherRiskResponse` - `200 OK`)
```json
{
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "timezone": "Asia/Kolkata"
  },
  "current": {
    "temperature_c": 21.2,
    "humidity_pct": 94.0,
    "precipitation_mm": 0.0,
    "wind_kmh": 6.4
  },
  "risk_level": "Critical",
  "risk_score": 0.925,
  "favorable_conditions_summary": "Smith Period criteria triggered: 14 consecutive hours with relative humidity >= 90% and temperatures between 15°C and 22°C. High infection and sporulation pressure for Late Blight / Downy Mildew.",
  "spray_advisory": "Optimal spray window in next 4-6 hours: Calm winds (<15 km/h) and dry conditions forecast. Favorable for protective or systemic fungicide/bactericide application."
}
```

---

## 6. Asynchronous Document Ingestion & Task Management

### `POST /api/v1/upload/async` (or `/upload/async`)

Queues large PDF documents (>10 pages) for non-blocking background ingestion, OCR recovery, CLIP image embedding, and vector indexing.

#### Request (Multipart Form)
- `file`: PDF file binary.
- `collection`: Optional named collection string (e.g. `corn-pathology-2026`).

#### Response (`AsyncTaskResponse` - `202 Accepted`)
```json
{
  "task_id": "task_9d21e847-2b01-44ca",
  "document_id": "doc_a1b2c3d4",
  "original_filename": "FAO_Tomato_Diseases_Manual.pdf",
  "status": "queued",
  "progress": 0.0,
  "message": "Document upload queued for background processing.",
  "created_at": 1723708800.0
}
```

---

### `GET /api/v1/documents/tasks/{task_id}`

Polls the ingestion progress and status transitions of an asynchronous task.

#### Status Lifecycle Transitions
`queued` $\to$ `extracting` $\to$ `chunking` $\to$ `embedding` $\to$ `indexing` $\to$ `completed` (or `failed`)

#### Response Schema (`TaskStatusResponse` - `200 OK`)
```json
{
  "task_id": "task_9d21e847-2b01-44ca",
  "document_id": "doc_a1b2c3d4",
  "original_filename": "FAO_Tomato_Diseases_Manual.pdf",
  "status": "completed",
  "progress": 100.0,
  "current_step": "Ingestion completed successfully.",
  "error": null,
  "result": {
    "document_id": "doc_a1b2c3d4",
    "original_filename": "FAO_Tomato_Diseases_Manual.pdf",
    "total_pages": 48,
    "total_chunks": 132,
    "total_embeddings": 132,
    "pages_ocred": 2,
    "total_images": 14,
    "images_captioned": 14,
    "images_embedded": 14,
    "total_tables": 4,
    "collection": "tomato-pathology-2026",
    "processing_time": 6.84,
    "status": "success"
  },
  "created_at": 1723708800.0,
  "updated_at": 1723708806.84
}
```

---

### `POST /api/v1/upload` (Synchronous Upload)
Synchronously parses, chunks, embeds, and indexes a PDF document, returning `DocumentProcessingResponse` (`201 Created`).

### `GET /api/v1/documents`
Lists all uploaded documents scoped to the caller's tenant. Supports `collection` filtering and admin `all_tenants=true` oversight.

### `DELETE /api/v1/documents/{document_id}`
Deletes a document from vector indexes (text + CLIP), removes extracted image files, and deletes durable metadata.

### `GET /api/v1/documents/{document_id}/file`
Serves the raw source PDF bytes for in-app PDF citation previews.

### `GET /api/v1/documents/{document_id}/pages/{page}/highlight`
Calculates PyMuPDF bounding box coordinate rectangles (`[x0, y0, x1, y1]`) for citation snippet text on a specific page.

---

## 7. Observability & Health Checks

### `GET /api/v1/health` (or `/health`)

Unauthenticated health and readiness probe reporting LLM provider connectivity, multimodal feature flags, vision service status, vector store type, and embedding cache statistics.

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "llm": {
    "provider": "gemini",
    "provider_configured": true,
    "fallback_provider": "groq",
    "fallback_configured": true,
    "model_routing_enabled": true
  },
  "multimodal": {
    "image_extraction_enabled": true,
    "image_captioning_enabled": true,
    "table_extraction_enabled": true,
    "vision_qa_enabled": true,
    "ocr_available": true
  },
  "vision_service": {
    "url": "http://localhost:8001",
    "online": true,
    "has_gemini_fallback": true,
    "disease_classes_count": 38
  },
  "vector_store": {
    "backend": "faiss",
    "model": "all-MiniLM-L6-v2",
    "status": "ready"
  },
  "cache": {
    "hits": 1420,
    "misses": 88,
    "hit_rate_pct": 94.16,
    "size": 1508
  },
  "database": "connected"
}
```

---

### `GET /api/v1/health/vision`
Dedicated liveness check for the LeafSense vision service (Port 8001).

---

### `GET /api/v1/metrics` (or `/metrics`)

Serves the in-process telemetry registry in standard **Prometheus text exposition format (version 0.0.4)**.

#### Authentication
Unauthenticated by default. If `METRICS_BEARER_TOKEN` is set, requires `Authorization: Bearer <METRICS_BEARER_TOKEN>`.

#### Sample Output (`text/plain; version=0.0.4`)
```prometheus
# HELP http_requests_total Total HTTP requests handled
# TYPE http_requests_total counter
http_requests_total{method="POST",path="/api/v1/chat/query",status_code="200"} 412
http_requests_total{method="POST",path="/api/v1/chat/diagnose",status_code="200"} 184

# HELP insightai_rag_requests_total Total RAG queries processed
# TYPE insightai_rag_requests_total counter
insightai_rag_requests_total{endpoint="/api/v1/query",status="success"} 398

# HELP insightai_rag_latency_seconds Latency of RAG query pipelines
# TYPE insightai_rag_latency_seconds histogram
insightai_rag_latency_seconds_bucket{le="0.5"} 42
insightai_rag_latency_seconds_bucket{le="1.0"} 298
insightai_rag_latency_seconds_bucket{le="2.0"} 410
insightai_rag_latency_seconds_bucket{le="+Inf"} 412
insightai_rag_latency_seconds_sum 584.21
insightai_rag_latency_seconds_count 412

# HELP insightai_total_vectors Total active embeddings in vector store
# TYPE insightai_total_vectors gauge
insightai_total_vectors 12480
```

---

## 8. Human-in-the-Loop Feedback

### `POST /api/v1/feedback` (or `/feedback`)

Records user ratings (`up` / `down`), comments, and optional 7-criteria quality rubrics for RAG answers.

#### Request Body (`FeedbackRequest`)
```json
{
  "message_id": "msg_89a2b1c4",
  "rating": "up",
  "comment": "Accurate dosage recommendation for Chlorothalonil.",
  "rubric": {
    "correctness": 5,
    "helpfulness": 5,
    "completeness": 5,
    "safety": 5,
    "tone": 5,
    "groundedness": 5,
    "citation_quality": 5
  }
}
```

---

## 9. Code Examples & SDK Usage

### 9.1 cURL

#### Querying Chat RAG
```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "X-API-Key: test_key_udbhav" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do you manage late blight on potatoes?",
    "top_k": 5
  }'
```

#### Uploading Leaf Photo for Diagnosis
```bash
curl -X POST "http://localhost:8000/api/v1/chat/diagnose" \
  -H "X-API-Key: test_key_udbhav" \
  -F "image=@/path/to/tomato_leaf.jpg" \
  -F "latitude=28.6139" \
  -F "longitude=77.2090"
```

### 9.2 JavaScript / TypeScript (fetch & EventSource)

#### Consuming the SSE Chat Stream
```javascript
async function streamAgroSenseChat(query, onChunk, onDone) {
  const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'test_key_udbhav',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));
        if (event.type === 'answer_chunk') {
          onChunk(event.payload.token || event.text);
        } else if (event.type === 'done') {
          onDone(event.payload);
        }
      }
    }
  }
}
```

### 9.3 Python (httpx)

#### Asynchronous Ingestion & Task Polling
```python
import httpx
import time

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "test_key_udbhav"}

def upload_and_await_ingestion(pdf_path: str):
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=60.0) as client:
        with open(pdf_path, "rb") as f:
            resp = client.post("/api/v1/upload/async", files={"file": f})
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        print(f"Queued async task: {task_id}")

        # Poll status
        while True:
            poll_resp = client.get(f"/api/v1/documents/tasks/{task_id}")
            data = poll_resp.json()
            status = data["status"]
            progress = data["progress"]
            print(f"Task status: {status} ({progress}%)")
            
            if status == "completed":
                print("Ingestion successful:", data["result"])
                return data["result"]
            elif status == "failed":
                raise RuntimeError(f"Ingestion failed: {data['error']}")
            
            time.sleep(1.0)
```
