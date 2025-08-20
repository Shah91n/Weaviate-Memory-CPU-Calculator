"""
Weaviate Memory & CPU Calculator
Based on official Weaviate documentation for resource planning
"""

from dataclasses import dataclass
from enum import Enum
import math

class CompressionType(Enum):
	NONE = "none"
	PQ = "pq" # Product Quantization: 85% reduction, requires training
	BQ = "bq" # Binary Quantization: 97% reduction, no training
	SQ = "sq" # Scalar Quantization: 75% reduction, requires training  
	RQ = "rq" # Rotational Quantization: 75% reduction, no training (v1.26+)

@dataclass
class ResourceEstimate:
	"""Container for resource estimation results"""
	memory_gb: float
	memory_gb_with_pq: float
	memory_gb_with_bq: float
	memory_gb_with_sq: float
	memory_gb_with_rq: float
	disk_storage_gb: float
	min_cpu_cores: int
	recommended_cpu_cores: int
	recommended_memory_gb: float
	vectors_memory_gb: float
	connections_memory_gb: float
	raw_vector_size_gb: float
	compression_savings_gb: float

class WeaviateResourceCalculator:
	"""
	Calculate Weaviate resource requirements based on official documentation.
	
	Uses formulas from: https://weaviate.io/developers/weaviate/concepts/resources
	"""

	def __init__(self):
		self.bytes_per_float32 = 4 # Standard float32 size
		self.bytes_per_connection = 10 # 8-10B per HNSW connection (using conservative 10B)
		self.gc_overhead_multiplier = 2.0 # 2x for garbage collection overhead

	def calculate_vector_memory(self, num_objects: int, dimensions: int) -> float:
		"""
		Calculate memory required for vectors only (without compression).
		
		Think of this as: How much space do I need to store all my data?
		Each number in a vector takes 4 bytes (float32)
		
		Formula: objects × dimensions × 4 bytes
		Returns: Memory in GB
		"""
		total_bytes = num_objects * dimensions * self.bytes_per_float32
		return total_bytes / (1024 ** 3) # Convert to GB

	def calculate_hnsw_connections_memory(self, num_objects: int, max_connections: int) -> float:
		"""
		Calculate memory required for HNSW connections (the "graph" part).
		
		HNSW builds a graph where each vector connects to nearby vectors.
		Think of it like a highway system connecting cities.
		
		Formula: objects × avg_connections × bytes_per_connection
		Note: More accurate calculation than 2x - base layer has 2×maxConnections, 
		      other layers have maxConnections, averaging to ~1.5x
		Returns: Memory in GB
		"""
		# More accurate calculation: base layer (2x) + upper layers (1x) ≈ 1.5x average
		avg_connections_per_object = max_connections * 1.5
		total_bytes = num_objects * avg_connections_per_object * self.bytes_per_connection
		return total_bytes / (1024 ** 3) # Convert to GB

	def calculate_total_memory(self, vectors_memory_gb: float, connections_memory_gb: float,
		compression: CompressionType = CompressionType.NONE) -> float:
		"""
		Calculate total memory including compression and GC overhead.
		
		Args:
		    vectors_memory_gb: Memory for vectors
		    connections_memory_gb: Memory for HNSW connections  
		    compression: Compression type
		    
		Returns: Total memory in GB
		"""
		# Apply compression to vectors only (connections are not compressed)
		if compression == CompressionType.PQ:
			compression_factor = 0.15 # 85% reduction
		elif compression == CompressionType.BQ:
			compression_factor = 0.03 # 97% reduction  
		elif compression == CompressionType.SQ:
			compression_factor = 0.25 # 75% reduction
		elif compression == CompressionType.RQ:
			compression_factor = 0.25 # 75% reduction (no training required)
		else:
			compression_factor = 1.0

		compressed_vectors_memory = vectors_memory_gb * compression_factor

		# Apply GC overhead to vector memory only
		vectors_with_gc = compressed_vectors_memory * self.gc_overhead_multiplier

		return vectors_with_gc + connections_memory_gb

	def calculate_disk_storage(self, num_objects: int, dimensions: int, 
		object_size_kb: float = 4.0, compression: CompressionType = CompressionType.NONE) -> float:
		"""
		Calculate disk storage requirements.
		
		Important: With compression, BOTH original and compressed vectors are stored on disk.
		This is because Weaviate uses "rescoring" - it searches compressed vectors first,
		then uses original vectors for final accuracy.
		
		Args:
		    num_objects: Number of objects
		    dimensions: Vector dimensions
		    object_size_kb: Average size of object metadata in KB
		    compression: Compression type
		    
		Returns: Disk storage in GB
		"""
		# Vector storage
		vector_bytes = num_objects * dimensions * self.bytes_per_float32
		vector_gb = vector_bytes / (1024 ** 3)

		# With compression, both original and compressed vectors are stored
		if compression != CompressionType.NONE:
			if compression == CompressionType.PQ:
				compressed_factor = 0.15
			elif compression == CompressionType.BQ:
				compressed_factor = 0.03
			elif compression == CompressionType.SQ:
				compressed_factor = 0.25
			elif compression == CompressionType.RQ:
				compressed_factor = 0.25
			else:
				compressed_factor = 1.0

			total_vector_gb = vector_gb + (vector_gb * compressed_factor)
		else:
			total_vector_gb = vector_gb

		# Object metadata storage (text, properties, etc.)
		metadata_bytes = num_objects * object_size_kb * 1024 # KB to bytes
		metadata_gb = metadata_bytes / (1024 ** 3)

		# Add 20% overhead for indexes and system data
		raw_storage = total_vector_gb + metadata_gb
		return raw_storage * 1.2

	def calculate_cpu_requirements(self, target_qps: int, expected_latency_ms: int) -> tuple:
		"""
		Calculate CPU requirements based on target QPS and latency.
		
		Based on official Weaviate documentation:
		- "Each insert, or search, is single-threaded"
		- "If you make multiple searches at the same time, Weaviate can make use of multiple threads"
		- "When search throughput is limited, add CPUs to increase the number of queries per second"
		
		Official formula from benchmark docs:
		"If Weaviate were single-threaded, the throughput per second would roughly equal to 
		 1s divided by mean latency. For example, with a mean latency of 5ms, this would 
		 mean that 200 requests can be answered in a second."
		
		Args:
		    target_qps: Target queries per second
		    expected_latency_ms: Expected query latency in milliseconds
		    
		Returns: (min_cores, recommended_cores)
		"""
		# Official Weaviate formula: QPS_per_core = 1000ms / latency_ms
		# This is the theoretical maximum for a single core
		theoretical_qps_per_core = 1000.0 / expected_latency_ms

		# Apply real-world efficiency factor
		# Based on Weaviate benchmarks showing ~70-90% efficiency in practice
		# due to synchronization mechanisms, locks, and other overhead
		efficiency_factor = 0.8
		realistic_qps_per_core = theoretical_qps_per_core * efficiency_factor

		# Calculate minimum cores needed
		min_cores = max(1, math.ceil(target_qps / realistic_qps_per_core))

		# Recommended cores: add headroom for:
		# 1. Import operations (also CPU-bound)
		# 2. Peak load handling
		# 3. Non-linear scaling due to contention
		recommended_cores = min_cores * 2

		return min_cores, recommended_cores

	def get_deployment_recommendation(self, num_objects: int, memory_gb: float) -> tuple:
		"""
		Get deployment recommendations based on scale.
		
		Returns: (deployment_type, instance_recommendation, notes)
		"""
		if num_objects < 100_000:
			return (
				"Single Docker Container", 
				"4-8 GB RAM, 2-4 CPU cores",
				"Perfect for development and small projects"
			)
		elif num_objects < 1_000_000:
			return (
				"Docker Compose", 
				"8-16 GB RAM, 4-8 CPU cores", 
				"Good for production with <1M vectors"
			)
		elif num_objects < 10_000_000:
			return (
				"Kubernetes (2-3 nodes)", 
				"32-64 GB RAM, 16-32 CPU cores per node",
				"Recommended for serious production workloads"
			)
		else:
			return (
				"Kubernetes Cluster (3+ nodes)", 
				"128+ GB RAM, 32+ CPU cores per node",
				"Enterprise-scale deployment with high availability"
			)

	def get_optimization_tips(self, num_objects: int, dimensions: int, max_connections: int, 
		memory_gb: float, compression: CompressionType) -> list:
		"""
		Generate specific optimization recommendations.
		"""
		tips = []

		# Memory optimization tips
		if memory_gb > 50 and compression == CompressionType.NONE:
			tips.append("💡 Consider enabling Product Quantization (PQ) to reduce memory by 85%")

		if memory_gb > 100 and compression in [CompressionType.NONE, CompressionType.SQ, CompressionType.RQ]:
			tips.append("💡 For extreme memory savings, try Binary Quantization (BQ) - 97% reduction")

		# HNSW optimization
		if max_connections > 32 and dimensions >= 768:
			tips.append("💡 Reduce maxConnections to 16-32 for high-dimensional vectors (768D+)")

		# Dimension optimization  
		if dimensions > 1536:
			tips.append("💡 Consider using a lower-dimensional embedding model if accuracy allows")

		# Scale-specific tips
		if num_objects > 1_000_000:
			tips.append("💡 Use multiple shards for better import performance on large datasets")

		if num_objects > 10_000_000:
			tips.append("💡 Consider horizontal scaling with multiple Weaviate nodes")

		return tips

	def get_recommended_resources(self, num_objects: int, dimensions: int,
		target_qps: int = 50, expected_latency_ms: int = 50,
		max_connections: int = 32, 
		compression: CompressionType = CompressionType.NONE,
		object_size_kb: float = 4.0) -> ResourceEstimate:
		"""
		Get complete resource recommendations for a Weaviate deployment.
		
		Args:
		    num_objects: Number of vectors/objects
		    dimensions: Vector dimensions
		    target_qps: Target queries per second
		    expected_latency_ms: Expected query latency
		    max_connections: HNSW maxConnections parameter
		    compression: Compression type
		    object_size_kb: Average object metadata size
		    
		Returns: ResourceEstimate with all calculations
		"""
		# Calculate base memory requirements
		vectors_memory_gb = self.calculate_vector_memory(num_objects, dimensions)
		connections_memory_gb = self.calculate_hnsw_connections_memory(num_objects, max_connections)

		# Calculate memory for different compression types
		memory_no_compression = self.calculate_total_memory(vectors_memory_gb, connections_memory_gb, CompressionType.NONE)
		memory_with_pq = self.calculate_total_memory(vectors_memory_gb, connections_memory_gb, CompressionType.PQ)
		memory_with_bq = self.calculate_total_memory(vectors_memory_gb, connections_memory_gb, CompressionType.BQ)
		memory_with_sq = self.calculate_total_memory(vectors_memory_gb, connections_memory_gb, CompressionType.SQ)
		memory_with_rq = self.calculate_total_memory(vectors_memory_gb, connections_memory_gb, CompressionType.RQ)

		# Calculate disk storage
		disk_storage_gb = self.calculate_disk_storage(num_objects, dimensions, object_size_kb, compression)

		# Calculate CPU requirements
		min_cpu_cores, recommended_cpu_cores = self.calculate_cpu_requirements(target_qps, expected_latency_ms)

		# Add 25% buffer to memory recommendation for safety
		recommended_memory_gb = memory_no_compression * 1.25

		# Calculate compression savings
		compression_savings_gb = memory_no_compression - memory_with_pq

		return ResourceEstimate(
			memory_gb=memory_no_compression,
			memory_gb_with_pq=memory_with_pq,
			memory_gb_with_bq=memory_with_bq,
			memory_gb_with_sq=memory_with_sq,
			memory_gb_with_rq=memory_with_rq,
			disk_storage_gb=disk_storage_gb,
			min_cpu_cores=min_cpu_cores,
			recommended_cpu_cores=recommended_cpu_cores,
			recommended_memory_gb=recommended_memory_gb,
			vectors_memory_gb=vectors_memory_gb,
			connections_memory_gb=connections_memory_gb,
			raw_vector_size_gb=vectors_memory_gb,
			compression_savings_gb=compression_savings_gb
		)

# Embedding models with latest 2025 offerings and dimensions
EMBEDDING_MODELS = {
	"OpenAI": {
		"text-embedding-3-large": 3072,
		"text-embedding-3-small": 1536
	},
	"Google Gemini": {
		"gemini-embedding-001": 3072,
		"text-multilingual-embedding-002": 768
	},
	"Cohere": {
		"embed-v4": 1536,
		"embed-multilingual-v3.0": 1024
	},
	"Anthropic/Voyage": {
		"voyage-large-2": 1536,
		"voyage-code-2": 1536,
		"voyage-2": 1024,
	},
	"Mistral": {
		"mistral-embed": 1024,
		"mistral-embed-large": 1024,
	},
	"Custom/Other": {
		"Custom dimensions": 768, # Default placeholder
	}
}
