typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;

task void compute_assignments_ispc_task(uniform WorkerArgs* uniform args,
                                            uniform int span,
                                            uniform double* uniform minDist){
  uniform int indexStart = taskIndex * span;
  uniform int indexEnd = min(args->M, indexStart + span);
  
  uniform int D = args->N; // 维度

  /*
  foreach (m = indexStart ... indexEnd) {
    // Initialize arrays
    double m_minDist = 1e30;
    int m_assignment = -1;

    // Assign datapoints to closest centroids
    for(uniform int k = args->start; k < args->end; k++) {
      double d2 = 0;

      // 遍历所有维度 D
      for(uniform int i = 0; i < D; i++) {
        double diff = args->data[m * D + i] - args->clusterCentroids[k * D + i];
        d2 += diff * diff;
      }
      d2 = sqrt(d2);
      if(d2 < m_minDist) {
        m_minDist = d2;
        m_assignment = k;
      }
    }
    minDist[m] = m_minDist;
    args->clusterAssignments[m] = m_assignment;
  }
  */
  
  for(uniform int m = indexStart; m < indexEnd; m++)
  {
    uniform double m_minDist = 1e30;
    uniform int m_assignment = -1;

    for(uniform int k = args->start; k < args->end; k++){
      // 在维度 D 上使用 foreach
      // 这会让 8 个通道读取内存中连续的 [d, d+1, ... d+7]
      // 从而触发硬件的 L1 Cache 预取和连续 Load 指令
      double d2 = 0;
      // 遍历所有维度 D
      foreach(i = 0 ... D) {
        double diff = args->data[m * D + i] - args->clusterCentroids[k * D + i];
        d2 += diff * diff;
      }
      
      // 将 8 个通道的局部和汇总
      uniform double total_d2 = sqrt(reduce_add(d2));

      if(total_d2 < m_minDist) {
        m_minDist = total_d2;
        m_assignment = k;
      }
    }
    minDist[m] = m_minDist;
    args->clusterAssignments[m] = m_assignment;
  }
  
}

export void compute_assignments_ispc_withtasks(uniform WorkerArgs* uniform args,
                                                uniform double* uniform minDist) {
   
  uniform int numTasks = 64;    // 64 tasks
  uniform int span = (args->M + numTasks - 1) / numTasks;

  launch[args->M/span] compute_assignments_ispc_task(args, span, minDist);

}