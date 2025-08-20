# Weaviate Memory & CPU Calculator

**🎯 Resource planning tool for Weaviate vector database deployments**

<a href="https://weaviate-memory-cpu-calculator.streamlit.app/">
  Visit Weaviate Memory & CPU Calculator Web App
</a>

Beginner-friendly calculator based on [official Weaviate documentation](https://weaviate.io/developers/weaviate/concepts/resources) to help you plan memory, CPU, and storage requirements for your vector database deployment.

## 🚀 Features

### ✅ **Calculations**
- **Memory formula** with 2x GC overhead
- **HNSW calculation** (1.5x vs 2x connections for better accuracy)
- **CPU formula** from Weaviate benchmark docs (1000ms ÷ latency × 80% efficiency)
- **All compression methods** supported (PQ, BQ, SQ, RQ)

### 🎓 **User-Friendly Interface**
- **Step-by-step breakdowns** for Memory, CPU, and Disk calculations
- **Visual charts** showing memory composition and compression savings
- **Optimization tips** based on your configuration

### 📊 **Comprehensive Planning**
- **Memory requirements** with and without compression
- **CPU calculations** based on target QPS and latency
- **Disk storage estimates**
- **Deployment recommendations** (Docker, Kubernetes, Cloud)

### 🗜️ **All Compression Methods**
- **Product Quantization (PQ)**: 85% memory reduction, best balance
- **Binary Quantization (BQ)**: 97% memory reduction, maximum savings
- **Scalar Quantization (SQ)**: 75% memory reduction, fast compression
- **Rotational Quantization (RQ)**: 75% memory reduction, no training required

## 📋 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Shah91n/Weaviate-Memory-CPU-Calculator.git
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

### 🔗 **Related Tools**
- **[Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)** - Detailed disk usage calculations
- **[Source Code](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)** - Comprehensive disk planning tool

## 📚 Documentation References

1. **[Resource Planning](https://weaviate.io/developers/weaviate/concepts/resources)** - Memory and CPU formulas
2. **[Vector Indexing](https://weaviate.io/developers/weaviate/concepts/vector-indexing)** - HNSW parameters
3. **[Compression Guide](https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression)** - All compression methods
4. **[ANN Benchmarks](https://weaviate.io/developers/weaviate/benchmarks/ann)** - CPU calculation source

## 🤝 Contributing

We welcome contributions!

## 📄 License

MIT License - See LICENSE file for details

**💡 Made for the community**

Help others plan their vector database deployments with confidence. Star ⭐ this repo if it helped you!
