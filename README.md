# Weaviate Memory & CPU Calculator v2.0

**🎯 Enterprise-grade resource planning tool for Weaviate vector database deployments**

An accurate, beginner-friendly calculator based entirely on [official Weaviate documentation](https://weaviate.io/developers/weaviate/concepts/resources) to help you plan memory, CPU, and storage requirements for your vector database deployment.

![Weaviate Calculator](https://img.shields.io/badge/Weaviate-v1.26+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![Python](https://img.shields.io/badge/Python-3.8+-green)

## 🚀 Features

### ✅ **Accurate Calculations**
- **Memory formulas** based on official Weaviate documentation with 2x GC overhead
- **Improved HNSW calculation** (1.5x vs 2x connections for better accuracy)
- **Official CPU formula** from Weaviate benchmark docs (1000ms ÷ latency × 80% efficiency)
- **All compression methods** supported (PQ, BQ, SQ, RQ)
- **Verified against 2025 documentation**

### 🎓 **User-Friendly Interface**
- **Step-by-step breakdowns** for Memory, CPU, and Disk calculations
- **Visual charts** showing memory composition and compression savings
- **Real-world examples** and deployment recommendations
- **Optimization tips** based on your configuration

### 📊 **Comprehensive Planning**
- **Memory requirements** with and without compression
- **CPU calculations** based on target QPS and latency using official Weaviate formulas
- **Disk storage estimates** including compression overhead
- **Deployment recommendations** (Docker, Kubernetes, Cloud)
- **Cost optimization tips** for production deployments

### 🗜️ **All Compression Methods**
- **Product Quantization (PQ)**: 85% memory reduction, best balance
- **Binary Quantization (BQ)**: 97% memory reduction, maximum savings
- **Scalar Quantization (SQ)**: 75% memory reduction, fast compression
- **Rotational Quantization (RQ)**: 75% memory reduction, no training required

### 🤖 **Latest 2025 Embedding Models**
- **OpenAI**: text-embedding-3-large (3072D), text-embedding-3-small (1536D)
- **Google Gemini**: gemini-embedding-001 (3072D), text-multilingual-embedding-002 (768D)
- **Cohere**: embed-v4 (1536D), embed-multilingual-v3.0 (1024D)
- **Anthropic/Voyage**: voyage-large-2 (1536D), voyage-2 (1024D)
- **Mistral**: mistral-embed (1024D)

## 📋 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd weaviate-calculator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open in browser:** http://localhost:8501

### Docker Deployment

```bash
# Build the image
docker build -t weaviate-calculator .

# Run the container
docker run -p 8501:8501 weaviate-calculator
```

## 🧮 How It Works

### Memory Calculation (Official Formula)

```python
# Rule of thumb (recommended)
memory = 2 × (objects × dimensions × 4 bytes) + hnsw_connections

# Detailed calculation
vector_memory = objects × dimensions × 4 bytes
connections_memory = objects × (maxConnections × 1.5) × 10 bytes
total_memory = (vector_memory × 2) + connections_memory
```

**Why 2x multiplier?** Accounts for Go's garbage collection overhead and temporary allocations during imports.

### CPU Requirements (Official Weaviate Formula)

```python
# From official Weaviate benchmark documentation:
# "If Weaviate were single-threaded, throughput would equal 1s ÷ mean latency"
theoretical_qps_per_core = 1000ms ÷ latency_ms

# Apply real-world efficiency (80% due to synchronization)
realistic_qps_per_core = theoretical_qps_per_core × 0.8

# Calculate cores needed
min_cores = target_qps ÷ realistic_qps_per_core
recommended_cores = min_cores × 2  # headroom for imports and peaks
```

### Compression Impact

| Method | Memory Reduction | Training Required | Best For |
|--------|------------------|-------------------|----------|
| **PQ** | 85% reduction | ✅ Yes | Production (best balance) |
| **BQ** | 97% reduction | ❌ No | Maximum savings |
| **SQ** | 75% reduction | ✅ Yes | Fast compression |
| **RQ** | 75% reduction | ❌ No | No-config compression |

## 📊 Example Configurations

| Use Case | Objects | Dimensions | Memory (No Compression) | Memory (PQ) | CPU Cores | Deployment |
|----------|---------|------------|-------------------------|-------------|-----------|------------|
| **Personal Blog** | 10K | 384 | 0.1 GB | 0.02 GB | 2 | Docker |
| **E-commerce** | 100K | 768 | 0.6 GB | 0.1 GB | 4 | Docker Compose |
| **Knowledge Base** | 1M | 1536 | 12.3 GB | 1.8 GB | 8 | Kubernetes |
| **Enterprise RAG** | 10M | 1536 | 123 GB | 18.5 GB | 16 | K8s Cluster |
| **Large-scale SaaS** | 100M | 3072 | 2457 GB | 368 GB | 32+ | Multi-region K8s |

## 🔧 Configuration Examples

### Environment Variables
```bash
# Limit Weaviate to 80% of available memory
LIMIT_RESOURCES=true

# Set Go memory limit (10-20% of total memory)
GOMEMLIMIT=2GB

# Set CPU threads
GOMAXPROCS=16
```

### HNSW Configuration
```json
{
  "vectorIndexConfig": {
    "maxConnections": 32,        // Optimal for 768D+ vectors
    "efConstruction": 128,       // Build quality
    "ef": 100,                   // Query quality
    "dynamicEfMin": 100,
    "dynamicEfMax": 500
  }
}
```

### Compression Setup
```json
{
  "vectorIndexConfig": {
    "pq": {
      "enabled": true,
      "trainingLimit": 100000,
      "segments": 96
    }
  }
}
```

## 🎯 Planning Guidelines

### Small Projects (< 100K vectors)
- **Deployment**: Single Docker container
- **Memory**: 4-8 GB RAM
- **CPU**: 2-4 cores
- **Compression**: Not needed
- **Storage**: < 1 GB

### Medium Projects (100K - 1M vectors)
- **Deployment**: Docker Compose
- **Memory**: 8-32 GB RAM
- **CPU**: 4-16 cores
- **Compression**: Consider PQ
- **Storage**: 1-50 GB

### Large Projects (1M+ vectors)
- **Deployment**: Kubernetes
- **Memory**: 32+ GB RAM
- **CPU**: 16+ cores
- **Compression**: Recommended (PQ or BQ)
- **Storage**: 50+ GB

## 💡 Key Features

### 📐 **Detailed Calculations**
The calculator shows step-by-step breakdowns for:
- **Memory**: Vector storage + HNSW connections + GC overhead
- **CPU**: Official Weaviate formula with real-world efficiency factors
- **Disk**: Basic calculation with link to specialized [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)

### 🔗 **Related Tools**
- **[Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)** - Detailed disk usage calculations
- **[Source Code](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)** - Comprehensive disk planning tool

### 🎯 **Optimization Tips**
- ✅ PQ compression for 85% memory savings
- ✅ BQ for maximum savings (97%) on high-dimensional data
- ✅ Reduce maxConnections to 16-32 for 768D+ vectors
- ✅ Multiple shards for better import performance
- ✅ Right-size deployment based on actual usage

## 📚 Documentation References

This calculator is based on official Weaviate documentation:

1. **[Resource Planning](https://weaviate.io/developers/weaviate/concepts/resources)** - Memory and CPU formulas
2. **[Vector Indexing](https://weaviate.io/developers/weaviate/concepts/vector-indexing)** - HNSW parameters
3. **[Compression Guide](https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression)** - All compression methods
4. **[ANN Benchmarks](https://weaviate.io/developers/weaviate/benchmarks/ann)** - CPU calculation source

### Key Formula Sources

> **"Memory usage = 2 × (the memory footprint of all vectors)"**  
> *— Weaviate Resource Planning Guide*

> **"If Weaviate were single-threaded, the throughput per second would roughly equal to 1s divided by mean latency"**  
> *— Weaviate ANN Benchmark Documentation*

> **"PQ compressed vectors typically use 85% less memory than uncompressed vectors. BQ compressed vectors use 97% less memory than uncompressed vectors."**  
> *— Weaviate Compression Documentation*

## 🤝 Contributing

We welcome contributions! This tool strictly follows Weaviate's official documentation.

### Guidelines
- All updates must reference specific documentation pages
- No assumptions - only use verified formulas
- Test calculations against real deployments
- Maintain accuracy and simplicity

## ⚠️ Important Notes

1. **Official Formula Compliance**: All calculations are based on verified Weaviate formulas
2. **Production Safety**: Includes recommended safety margins and real-world efficiency factors
3. **Compression Trade-offs**: Understand accuracy vs memory trade-offs before choosing compression
4. **Performance Validation**: Always test with your actual data before production deployment

## 🏷️ Version History

### v2.0 (Current)
- ✅ Improved HNSW calculation accuracy (1.5x vs 2x connections)
- ✅ Official CPU calculation using Weaviate benchmark formula
- ✅ Added detailed step-by-step calculation breakdowns
- ✅ Added RQ compression support (Rotational Quantization)
- ✅ Updated 2025 embedding model dimensions
- ✅ Added deployment and optimization recommendations
- ✅ Verified all formulas against latest Weaviate documentation

### v1.0
- Basic memory and CPU calculations
- PQ, BQ, SQ compression support
- HNSW parameter configuration
- Embedding model presets

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Weaviate Team** for excellent documentation and open-source vector database
- **Community Contributors** for feedback and real-world validation
- **Open Source Community** for making tools like this possible

---

**💡 Made for the community by the community**

Help others plan their vector database deployments with confidence. Star ⭐ this repo if it helped you!

**🔗 For detailed disk storage calculations, use the specialized [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)**
