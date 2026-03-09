#ifndef _TASKSYS_H
#define _TASKSYS_H

#include <thread>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <unordered_set>
#include "itasksys.h"

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        // 线程数量
        int num_threads;
        // 线程池
        std::vector<std::thread> thread_pool;
        // 是否完成
        std::atomic<bool> done{false};

        // 批次任务的状态
        class task{
            public:
                IRunnable* runnable;
                // 当前批次任务ID
                TaskID task_id;
                // 当前批次任务数量
                int total_tasks;
                // 下一个任务下标
                std::atomic<int> next_task_idx{0};
                // 已经完成的任务数量
                std::atomic<int> completed_tasks{0};
                task(IRunnable* runnable_, TaskID task_id_, int total_tasks_)
                    : runnable(runnable_), task_id(task_id_), total_tasks(total_tasks_){}
        };

        // 同步原语
        std::mutex mtx;
        std::condition_variable cv_worker; // 唤醒工作线程
        std::condition_variable cv_main;   // 唤醒主线程

        TaskID total_task_id{0};    // 总共批次数量
        std::unordered_map<TaskID,std::vector<TaskID>> deps;                   // 依赖关系<A,B>，只有完成A后，才能执行B
        std::unordered_map<TaskID,int> deps_total;             // task_id的依赖数量
        std::unordered_map<TaskID,std::shared_ptr<task>> id_task;               // task_id对应的任务，目前不可以运行的任务
        std::queue<std::shared_ptr<task>> run_queue;    // 当前可以执行的任务队列
        std::unordered_set<TaskID> id_finished;    // task_id对应的任务已经完成

};

#endif
