# CBIR API - cURL Commands for Postman Testing

Base URL: `http://localhost:8000`

Copy each command and paste into Postman's "Import > Raw text" or run directly in terminal.

---

## 1. Health Check

```bash
curl -X GET "http://localhost:8000/api/health" -H "Accept: application/json"
```

**Expected:** Returns server status, MongoDB connection, loaded features.

---

## 2. List Available Descriptors

```bash
curl -X GET "http://localhost:8000/api/descriptors" -H "Accept: application/json"
```

**Expected:** Returns list of descriptors with dimensions, metrics, and index types.

---

## 3. Browse Images (Paginated)

```bash
# Page 1, 20 images per page
curl -X GET "http://localhost:8000/api/images?page=1&page_size=20" -H "Accept: application/json"

# Page 2, 40 images per page
curl -X GET "http://localhost:8000/api/images?page=2&page_size=40" -H "Accept: application/json"
```

**Expected:** Returns paginated list of images with thumbnails and keywords.

---

## 4. Get Single Image Info

```bash
curl -X GET "http://localhost:8000/api/images/42" -H "Accept: application/json"
```

**Expected:** Returns metadata for image ID 42.

---

## 5. Text Search (AND mode)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["sunset", "beach"],
    "keyword_mode": "and",
    "k": 20
  }'
```

**Expected:** Returns images containing BOTH "sunset" AND "beach" keywords.

---

## 6. Text Search (OR mode)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["sunset", "beach"],
    "keyword_mode": "or",
    "k": 20
  }'
```

**Expected:** Returns images containing "sunset" OR "beach" (more results).

---

## 7. Visual Search - VGG16 (Euclidean)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns 20 visually similar images to image ID 0 using VGG16 deep features.

---

## 8. Visual Search - VGG16 (Cosine)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "cosine"},
    "index_type": "faiss",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns similar images using cosine similarity.

---

## 9. Visual Search - Edge Histogram (Euclidean)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["edgehistogram"],
    "metrics": {"edgehistogram": "euclidean"},
    "index_type": "faiss",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns similar images based on edge patterns.

---

## 10. Visual Search - Edge Histogram (Chi2)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["edgehistogram"],
    "metrics": {"edgehistogram": "chi2"},
    "index_type": "brute",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns similar images using Chi-squared distance (better for histograms).

---

## 11. Visual Search - Homogeneous Texture

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["homogeneoustexture"],
    "metrics": {"homogeneoustexture": "euclidean"},
    "index_type": "faiss",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns similar images based on texture patterns.

---

## 12. Visual Search - Brute Force (Reference)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "brute",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Same results as FAISS but slower (exhaustive search).

---

## 13. Multi-Descriptor Fusion (VGG16 + Edge Histogram)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16", "edgehistogram"],
    "metrics": {"vgg16": "euclidean", "edgehistogram": "euclidean"},
    "weights": {"vgg16": 0.7, "edgehistogram": 0.3},
    "index_type": "brute",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns images similar by BOTH semantic (VGG16) and structural (Edge) features.

---

## 14. Combined Search (Text + Visual)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["sunset"],
    "keyword_mode": "and",
    "query_image_id": 100,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 20,
    "exclude_query": true
  }'
```

**Expected:** Returns images matching "sunset" keyword, ranked by visual similarity to image 100.

---

## 15. Image Upload Search (External Query Image)

```bash
# Windows PowerShell
curl -X POST "http://localhost:8000/api/search/upload" `
  -H "Accept: application/json" `
  -F "file=@C:\path\to\your\image.jpg" `
  -F "descriptors=vgg16" `
  -F "metric=euclidean" `
  -F "index_type=faiss" `
  -F "k=20"

# Linux/Mac/Git Bash
curl -X POST "http://localhost:8000/api/search/upload" \
  -H "Accept: application/json" \
  -F "file=@/path/to/your/image.jpg" \
  -F "descriptors=vgg16" \
  -F "metric=euclidean" \
  -F "index_type=faiss" \
  -F "k=20"
```

**Expected:** Extracts VGG16 features from uploaded image and returns similar images.

---

## 16. Different Query Images

Test with different image IDs (0-999999):

```bash
# Image ID 500
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 500,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 10
  }'

# Image ID 10000
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 10000,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 10
  }'
```

---

## 17. Different Result Sizes (k values)

```bash
# k=5 (few results)
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 5
  }'

# k=50 (many results)
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 50
  }'

# k=100 (maximum useful results)
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 100
  }'
```

---

## 18. API Root Endpoint

```bash
curl -X GET "http://localhost:8000/" -H "Accept: application/json"
```

**Expected:** Returns API information and available endpoints.

---

## 19. OpenAPI Documentation

```bash
# Swagger UI (interactive)
# Open in browser: http://localhost:8000/docs

# ReDoc (alternative documentation)
# Open in browser: http://localhost:8000/redoc
```

---

## 20. Error Cases

### Invalid Image ID

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 9999999,
    "descriptors": ["vgg16"],
    "k": 20
  }'
```

**Expected:** Returns 404 error (image not found).

### Invalid Descriptor

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["invalid_descriptor"],
    "k": 20
  }'
```

**Expected:** Returns 400/500 error.

### Empty Keywords

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": [],
    "k": 20
  }'
```

**Expected:** Returns error (no search parameters provided).

---

## Testing Checklist for Report Screenshots

1. [ ] Health check showing "ok" status
2. [ ] Descriptors list showing all 3 descriptors
3. [ ] Image browse with pagination
4. [ ] Text search AND mode results
5. [ ] Text search OR mode results
6. [ ] Visual search VGG16 results
7. [ ] Visual search Edge Histogram results
8. [ ] Visual search Homogeneous Texture results
9. [ ] Multi-descriptor fusion results
10. [ ] Combined search (text + visual) results
11. [ ] Image upload search results
12. [ ] Error case handling

---

## Performance Testing

Run multiple searches and note response times:

```bash
# Time the response
curl -w "\nTime: %{time_total}s\n" -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_id": 0,
    "descriptors": ["vgg16"],
    "index_type": "faiss",
    "k": 20
  }'
```

---

## Notes

- All POST requests require `Content-Type: application/json`
- Image IDs range from 0 to 999999
- k values should be between 1 and 1000
- For upload, only `multipart/form-data` is accepted
- Only VGG16 is supported for uploaded images
