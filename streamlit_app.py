"""
Weaviate Memory & CPU Calculator
Resource planning tool based on official Weaviate documentation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
from weaviate_calculator import (
	WeaviateResourceCalculator, 
	CompressionType, 
	EMBEDDING_MODELS,
	ResourceEstimate
)

logo_path = os.path.join("assets", "weaviate-logo.png")
logo_image = Image.open(logo_path)

# Page configuration
st.set_page_config(
	page_title="Weaviate Resource Calculator",
	page_icon=logo_image,
	layout="wide",
	initial_sidebar_state="expanded"
)

# Initialize session state
if 'calculator' not in st.session_state:
	st.session_state.calculator = WeaviateResourceCalculator()

if 'results' not in st.session_state:
	st.session_state.results = None

def format_number(num: float, decimals: int = 2) -> str:
	"""Format number with appropriate units"""
	if num >= 1e9:
		return f"{num/1e9:.{decimals}f}B"
	elif num >= 1e6:
		return f"{num/1e6:.{decimals}f}M"
	elif num >= 1e3:
		return f"{num/1e3:.{decimals}f}K"
	else:
		return f"{num:.{decimals}f}"

def main():
	st.title("📟 Weaviate Memory & CPU Calculator")
	st.markdown("""
	            **Resource planning tool** based on [official Weaviate documentation](https://weaviate.io/developers/weaviate/concepts/resources)
	            """)

	# Create tabs
	tab1, tab2, tab3 = st.tabs(["📊 Calculator", "📖 How It Works & Guide", "📚 References"])

	with tab1:
		calculator_tab()

	with tab2:
		how_it_works_tab()

	with tab3:
		references_tab()

def calculator_tab():
	"""Main calculator interface"""
	col1, col2 = st.columns([1, 2])

	with col1:
		st.subheader("📥 Please fill in the following parameters")

		# Basic parameters
		num_objects = st.number_input(
			"Number of Objects/Vectors",
			min_value=10000,
			max_value=100_000_000_000,
			value=1_000_000,
			step=100_000,
			help="Total number of objects/vectors to store in Weaviate"
		)

		# Vector dimensions
		st.markdown("**Vector Dimensions**")
		dimension_source = st.radio(
			"Select dimension source:",
			["Embedding Model", "Custom"],
			horizontal=True
		)

		if dimension_source == "Embedding Model":
			provider = st.selectbox("Provider", list(EMBEDDING_MODELS.keys()))
			model = st.selectbox("Model", list(EMBEDDING_MODELS[provider].keys()))
			dimensions = EMBEDDING_MODELS[provider][model]
			st.info(f"**{dimensions} dimensions** for {model}")
		else:
			dimensions = st.number_input(
				"Custom Dimensions",
				min_value=1,
				max_value=10000,
				value=768,
				help="Number of dimensions per vector"
			)

		# Performance targets
		st.markdown("**Performance Requirements**")
		target_qps = st.number_input(
			"Target Queries Per Second (QPS)",
			min_value=1,
			max_value=10000,
			value=50,
			help="Expected query throughput"
		)

		expected_latency = st.slider(
			"Expected Query Latency (ms)",
			min_value=10,
			max_value=1000,
			value=50,
			step=10,
			help="Expected time per query in milliseconds"
		)

		# Advanced settings
		with st.expander("⚙️ Advanced Settings", expanded=False):
			st.markdown("**HNSW Index Configuration**")
			max_connections = st.slider(
				"maxConnections",
				min_value=4,
				max_value=128,
				value=32,
				step=4,
				help="HNSW graph connections per node. Lower values reduce memory but may impact recall."
			)

			st.markdown("**Compression**")
			compression_type = st.selectbox(
				"Vector Compression",
				["None", "Product Quantization (PQ)", "Binary Quantization (BQ)", "Scalar Quantization (SQ)", "Rotational Quantization (RQ)"],
				help="Choose compression method to reduce memory usage"
			)

			# Map selection to enum
			compression_map = {
				"None": CompressionType.NONE,
				"Product Quantization (PQ)": CompressionType.PQ,
				"Binary Quantization (BQ)": CompressionType.BQ,
				"Scalar Quantization (SQ)": CompressionType.SQ,
				"Rotational Quantization (RQ)": CompressionType.RQ
			}
			selected_compression = compression_map[compression_type]

			if compression_type != "None":
				if compression_type == "Product Quantization (PQ)":
					st.info("💡 PQ reduces memory usage to ~15% of original size (85% reduction)")
				elif compression_type == "Binary Quantization (BQ)":
					st.info("💡 BQ reduces memory usage to ~3% of original size (97% reduction)")
				elif compression_type == "Scalar Quantization (SQ)":
					st.info("💡 SQ reduces memory usage to ~25% of original size (75% reduction)")
				elif compression_type == "Rotational Quantization (RQ)":
					st.info("💡 RQ reduces memory usage to ~25% of original size (75% reduction)")

			object_size_kb = st.number_input(
				"Average Object Metadata Size (KB)",
				min_value=0.1,
				max_value=100.0,
				value=4.0,
				step=0.5,
				help="Average size of non-vector data per object"
			)

		# Calculate button
		if st.button("🔄 Calculate Resources", type="primary", use_container_width=True):
			results = st.session_state.calculator.get_recommended_resources(
				num_objects=int(num_objects),
				dimensions=dimensions,
				target_qps=target_qps,
				expected_latency_ms=expected_latency,
				max_connections=max_connections,
				compression=selected_compression
			)
			st.session_state.results = results
			st.session_state.current_params = {
				'num_objects': num_objects,
				'dimensions': dimensions,
				'max_connections': max_connections,
				'compression': selected_compression,
				'target_qps': target_qps,
				'expected_latency': expected_latency
			}

	with col2:
		if st.session_state.results:
			display_results(st.session_state.results, st.session_state.current_params)
		else:
			st.info("👈 Configure parameters and click 'Calculate Resources' to see results")

def display_results(results: ResourceEstimate, params: dict):
	"""Display calculation results"""
	st.subheader("📊 Resource Requirements")

	# Key metrics
	col1, col2, col3, col4 = st.columns(4)

	with col1:
		st.metric(
			"Memory (No Compression)",
			f"{results.memory_gb:.1f} GB",
			help="Using 2x rule of thumb for GC overhead"
		)

	with col2:
		st.metric(
			"Memory (With PQ)",
			f"{results.memory_gb_with_pq:.1f} GB",
			f"-{((1 - results.memory_gb_with_pq/results.memory_gb) * 100):.0f}%",
			help="85% reduction with Product Quantization"
		)

	with col3:
		st.metric(
			"Disk Storage",
			f"{results.disk_storage_gb:.1f} GB",
			help="Including 20% overhead for indexes"
		)

	with col4:
		st.metric(
			"Min CPU Cores",
			f"{results.min_cpu_cores}",
			help="For target QPS with efficiency factor"
		)

	# Compression comparison
	st.markdown("---")
	st.subheader("🗜️ Compression Options")

	compression_data = {
		"Method": ["No Compression", "Product Quantization (PQ)", "Binary Quantization (BQ)", "Scalar Quantization (SQ)", "Rotational Quantization (RQ)"],
		"Memory (GB)": [results.memory_gb, results.memory_gb_with_pq, results.memory_gb_with_bq, results.memory_gb_with_sq, results.memory_gb_with_rq],
		"Reduction": ["0%", "85%", "97%", "75%", "75%"],
		"Training Required": ["❌", "✅", "❌", "✅", "❌"],
		"Notes": ["Full precision", "Best balance", "Maximum savings", "Fast compression", "No training needed"]
	}

	compression_df = pd.DataFrame(compression_data)
	st.dataframe(compression_df, use_container_width=True)

	# Memory breakdown
	st.markdown("---")
	st.subheader("💾 Memory Breakdown")

	col1, col2 = st.columns(2)

	with col1:
		# Memory composition chart
		fig = go.Figure(data=[
			go.Bar(
				name='Vectors',
				x=['No Compression', 'With PQ'],
				y=[results.vectors_memory_gb * 2, results.vectors_memory_gb * 0.15 * 2],
				text=[f"{results.vectors_memory_gb * 2:.1f} GB", 
					f"{results.vectors_memory_gb * 0.15 * 2:.1f} GB"],
				textposition='auto',
			),
			go.Bar(
				name='HNSW Connections',
				x=['No Compression', 'With PQ'],
				y=[results.connections_memory_gb, results.connections_memory_gb],
				text=[f"{results.connections_memory_gb:.1f} GB", 
					f"{results.connections_memory_gb:.1f} GB"],
				textposition='auto',
			)
		])

		fig.update_layout(
			title="Memory Composition",
			yaxis_title="Memory (GB)",
			barmode='stack',
			height=400
		)
		st.plotly_chart(fig, use_container_width=True)

	with col2:
		# Calculation details
		st.markdown("**📐 Calculation Details**")

		st.code(f"""
		             # Vector Memory (No Compression)
		             Dimensions: {params['dimensions']}
		             Objects: {format_number(params['num_objects'])}
		             Bytes per vector: {params['dimensions']} × 4 = {params['dimensions'] * 4:,} bytes
		             Total vector memory: {results.vectors_memory_gb:.2f} GB
		             With GC overhead (2x): {results.vectors_memory_gb * 2:.2f} GB
		         
		             # HNSW Connections Memory
		             Max connections: {params['max_connections']}
		             Avg connections: {params['max_connections'] * 1.5:.0f} connections
		             Bytes per connection: 10
		             Total connections memory: {results.connections_memory_gb:.2f} GB
		         
		             # Total Memory
		             Rule of thumb: {results.memory_gb:.2f} GB
		             Recommended (+25%): {results.recommended_memory_gb:.2f} GB
		         """, language="python")

		st.markdown("**⚡ CPU Calculation**")

		target_qps = params['target_qps']
		expected_latency = params['expected_latency']

		theoretical_qps_per_core = 1000.0 / expected_latency
		realistic_qps_per_core = theoretical_qps_per_core * 0.8

		st.code(f"""
		             # CPU Requirements
		             Target QPS: {target_qps}
		             Expected latency: {expected_latency}ms
		             Theoretical QPS/core: 1000ms ÷ {expected_latency}ms = {theoretical_qps_per_core:.1f}
		             Real-world QPS/core: {theoretical_qps_per_core:.1f} × 0.8 = {realistic_qps_per_core:.1f}
		             Min cores needed: {target_qps} ÷ {realistic_qps_per_core:.1f} = {results.min_cpu_cores}
		             Recommended: {results.min_cpu_cores} × 2 = {results.recommended_cpu_cores} cores
		         """, language="python")

		st.markdown("**💿 Disk Storage Calculation**")

		st.code(f"""
		             # Basic Disk Storage
		             Vector storage: {results.vectors_memory_gb:.2f} GB
		             Metadata: {format_number(params['num_objects'])} × 4KB = {(params['num_objects'] * 4 / 1024 / 1024):.2f} GB
		             System overhead (20%): +{results.disk_storage_gb * 0.2:.2f} GB
		             Total disk: {results.disk_storage_gb:.2f} GB
		         
		             Note: With compression, both original + compressed stored
		         """, language="python")

		st.info("🔗 **For detailed disk calculations:** [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/) | [Source](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)")

	# Recommendations
	st.markdown("---")
	st.subheader("🎯 Deployment Recommendations")

	deployment_type, instance_rec, notes = st.session_state.calculator.get_deployment_recommendation(
		params['num_objects'], results.memory_gb
	)

	col1, col2 = st.columns(2)
	with col1:
		st.info(f"**Recommended Deployment:** {deployment_type}")
	with col2:
		st.info(f"**Instance Sizing:** {instance_rec}")

	# Cost optimization tips
	optimization_tips = st.session_state.calculator.get_optimization_tips(
		params['num_objects'], params['dimensions'], params['max_connections'],
		results.memory_gb, params['compression']
	)

	if optimization_tips:
		st.markdown("### 💡 Optimization Tips")
		for tip in optimization_tips:
			st.markdown(tip)

def how_it_works_tab():
	"""Combined how it works and guide"""
	st.header("📖 How It Works & Planning Guide")

	# Quick guide for beginners
	st.markdown("""
	            ### 🎯 Quick Planning Guide
	            
	            **New to vector databases?** Here's what you need to know:
	            
	            - **Vectors** = Lists of numbers representing your data's meaning
	            - **Dimensions** = How many numbers in each vector (more = better accuracy, more memory)
	            - **HNSW** = The search algorithm that connects similar vectors for fast searching
	            - **Compression** = Reduces memory usage but may reduce accuracy slightly
	            """)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
		            #### 🔢 Understanding Dimensions
		            
		            | Dimensions | Use Case | Memory Impact |
		            |------------|----------|---------------|
		            | 384 | Small projects | Low |
		            | 768 | General purpose | Medium |
		            | 1536 | High accuracy | High |
		            | 3072 | Maximum quality | Very High |
		            
		            **Rule:** More dimensions = better accuracy but more memory needed
		            """)

	with col2:
		st.markdown("""
		            #### 🗜️ Compression Guide
		            
		            | Method | Memory Saved | When to Use |
		            |--------|--------------|-------------|
		            | PQ | 85% | Best balance |
		            | SQ | 75% | Fast setup |
		            | BQ | 97% | Maximum savings |
		            | RQ | 75% | No training |
		            
		            **Rule:** Always consider compression for >5M vectors
		            """)

	st.markdown("""
	            ---
	            ### 🧮 How Calculations Work
	                
	            #### Memory Rule of Thumb
	            ```
	            Memory = 2 × (Number of vectors × Dimensions × 4 bytes) + HNSW connections
	            ```
	            
	            **The 2x multiplier accounts for:**
	            - Go's garbage collection overhead during imports
	            - Temporary memory allocations
	            - Safety buffer for production stability
	            
	            #### Detailed Memory Formula
	            ```python
	            # Vector memory
	            vector_memory = objects × dimensions × 4 bytes
	            
	            # HNSW connections memory
	            # Base layer: 2 × maxConnections, Upper layers: 1 × maxConnections
	            # Average: ~1.5 × maxConnections
	            connections_memory = objects × (maxConnections × 1.5) × 10 bytes
	            
	            # Total memory with GC overhead
	            total_memory = (vector_memory × 2) + connections_memory
	            ```
	            
	            #### CPU Requirements
	            ```python
	            # Official Weaviate formula from benchmark documentation
	            theoretical_qps_per_core = 1000ms ÷ query_latency_ms
	            
	            # Apply real-world efficiency (80% due to synchronization overhead)
	            realistic_qps_per_core = theoretical_qps_per_core × 0.8
	            
	            # Calculate cores needed
	            min_cores = target_qps ÷ realistic_qps_per_core
	            recommended_cores = min_cores × 2  # headroom for imports and peaks
	            ```
	            
	            **Key facts from Weaviate docs:**
	            - Each search is single-threaded, but multiple searches use multiple threads
	            - "When search throughput is limited, add CPUs to increase QPS"
	            - Import operations are also CPU-bound (building HNSW index)
	            - Real-world efficiency is ~80% due to synchronization mechanisms
	            
	            #### Storage Calculation
	            ```python
	            # With compression, both original and compressed vectors stored
	            vector_storage = objects × dimensions × 4 bytes
	            if compression_enabled:
	                total_vectors = original_vectors + compressed_vectors
	            
	            total_storage = (vectors + metadata) × 1.2  # 20% overhead
	            ```
	            """)
	st.info("🔗 **For detailed disk calculations:** [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/) | [Source](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)")

	# Step by step planning
	st.markdown("""
	            ---
	            ### 📋 Step-by-Step Planning
	            
	            1. **Count Your Data**: How many documents/items do you have?
	            2. **Choose Embedding Model**: Pick based on accuracy vs speed needs
	            3. **Calculate Base Memory**: Objects × Dimensions × 4 bytes
	            4. **Add Safety Buffer**: Multiply by 2 for garbage collection
	            5. **Add HNSW Memory**: For the search graph connections
	            6. **Consider Compression**: 85% savings with PQ, 97% with BQ
	            7. **Plan Deployment**: Docker for <1M, Kubernetes for >1M objects
	            
	            ### 💡 Best Practices
	            
	            **Memory Optimization:**
	            - Use compression for datasets >5M objects
	            - Reduce maxConnections for high-dimensional vectors (768D+)
	            - Consider lower-dimensional models when possible
	            
	            **Performance Tuning:**
	            - Multiple shards improve import speed
	            - Increase efConstruction/ef with lower maxConnections
	            - Use SSDs for storage (required for good performance)
	            
	            **Cost Optimization:**
	            - PQ compression can reduce cloud costs by 80%+
	            - Right-size deployment based on actual usage
	            - Use LIMIT_RESOURCES=true to prevent OOM kills
	            """)

def references_tab():
	"""Updated references with latest documentation"""
	st.header("📚 Documentation References")

	st.markdown("""
	            All calculations in this tool are based on **official Weaviate documentation**:
	            
	            ### Primary Sources
	            
	            1. **[Resource Planning Guide](https://weaviate.io/developers/weaviate/concepts/resources)** - Main memory and CPU formulas
	            2. **[Vector Indexing](https://weaviate.io/developers/weaviate/concepts/vector-indexing)** - HNSW parameter details
	            3. **[Compression Guide](https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression)** - All compression methods
	            4. **[Vector Quantization](https://weaviate.io/developers/weaviate/concepts/vector-quantization)** - Technical compression details
	            
	            ### Key Formulas Verification
	            
	            #### Memory Rule of Thumb ✅
	            > **"Memory usage = 2 × (the memory footprint of all vectors)"**  
	            > *Source: Official Weaviate Resource Planning Guide*
	            
	            #### HNSW Connections ✅  
	            > **"Each object in memory has at most maxConnections connections per layer. Each of the connections uses 8-10B of memory. Note that the base layer allows for 2 * maxConnections."**  
	            > *Source: Official Weaviate Resource Planning Guide*
	            
	            #### Compression Percentages ✅
	            > **"PQ compressed vectors typically use 85% less memory than uncompressed vectors. SQ compressed vectors use 75% less memory than uncompressed vectors. BQ compressed vectors use 97% less memory than uncompressed vectors."**  
	            > *Source: Official Weaviate Compression Documentation*
	            
	            ### 🔧 Configuration Examples
	            
	            #### Environment Variables
	            ```bash
	            # Limit Weaviate to 80% of available memory
	            LIMIT_RESOURCES=true
	            
	            # Set Go memory limit (10-20% of total memory for Weaviate)
	            GOMEMLIMIT=2GB
	            
	            # Set maximum CPU threads
	            GOMAXPROCS=16
	            ```
	            
	            #### HNSW Configuration
	            ```json
	            {
	              "vectorIndexConfig": {
	                "maxConnections": 32,        // Reduced for high-dimensional vectors
	                "efConstruction": 128,       // Build-time quality
	                "ef": 100,                   // Query-time quality
	                "dynamicEfMin": 100,
	                "dynamicEfMax": 500,
	                "dynamicEfFactor": 8
	              }
	            }
	            ```
	            
	            ### 🚀 Deployment Guides
	            
	            - [Docker Compose Setup](https://weaviate.io/developers/weaviate/installation/docker-compose)
	            - [Kubernetes Deployment](https://weaviate.io/developers/weaviate/installation/kubernetes)  
	            - [Weaviate Cloud](https://console.weaviate.cloud)
	            - [Environment Variables](https://weaviate.io/developers/weaviate/config-refs/env-vars)
	            
	            ---
	            
	            💡 **Note:** This calculator closely follows formulas from Weaviate's official documentation. 
	            Results should be accurate for most use cases, but for critical deployments, consider testing and benchmarking with your own data.
	            """)

if __name__ == "__main__":
	main()
