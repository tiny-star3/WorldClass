#include "tasksys.h"

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) { this->num_threads = num_threads; }
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::thread* threads = new std::thread[num_threads-1];

    // static assignment of tasks to threads
    // for (int i = 0; i < num_threads-1; i++) {
    //     threads[i] = std::thread([&](int task_begin, int task_end){
    //         for(int j = task_begin; j < task_end; j++)
    //         {
    //             runnable->runTask(j, num_total_tasks);
    //         }
    //     }, i * (num_total_tasks / num_threads), (i + 1) * (num_total_tasks / num_threads));
    // }
    
    // for(int j = (num_threads - 1) * (num_total_tasks / num_threads); j < std::min(num_threads * (num_total_tasks / num_threads), num_total_tasks); j++)
    // {
    //     runnable->runTask(j, num_total_tasks);
    // }


    // dynamic assignment of tasks to threads
    std::mutex* mutex = new std::mutex();
    int counter = 0;
    for(int i = 0; i < num_threads-1; i++) {
        threads[i] = std::thread([&](){
            int local_i = 0;
            while(true)
            {
                {
                    std::unique_lock<std::mutex> lk(*mutex);
                    if(counter>=num_total_tasks) return;
                    local_i = counter++;
                }
                runnable->runTask(local_i, num_total_tasks);
            }
        });
    }

    int local_i = 0;
    while(true)
    {
        {
            std::unique_lock<std::mutex> lk(*mutex);
            if(counter>=num_total_tasks) break;
            local_i = counter++;
        }
        runnable->runTask(local_i, num_total_tasks);
    }

    for (int i = 0; i < num_threads-1; i++) {
        threads[i].join();
    }

    delete mutex;
    delete[] threads;
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for(int i = 0; i < num_threads-1; i++)
    {
        thread_pool.emplace_back(std::thread([this](){
            while (!done.load()) {
                int total = total_tasks.load();
                
                if (next_task_idx.load() < total) {
                    int local_task = next_task_idx.fetch_add(1);
                    if (local_task < total) {
                        current_runnable->runTask(local_task, total);
                        completed_tasks.fetch_add(1);
                    }
                }
            }
        }));
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    done.store(true);
    for(int i = 0; i < num_threads-1; i++) thread_pool[i].join();
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    current_runnable = runnable;
    next_task_idx.store(0);
    completed_tasks.store(0);
    total_tasks.store(num_total_tasks);

    int local_task;
    while((local_task = next_task_idx.fetch_add(1)) < num_total_tasks)
    {
        runnable->runTask(local_task, num_total_tasks);
        completed_tasks.fetch_add(1);
    }

    while(completed_tasks.load() < num_total_tasks){ }

    total_tasks.store(0);
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    for(int i = 0; i < num_threads-1; i++)
    {
        thread_pool.emplace_back(std::thread([this](){
            while (true) {
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv_worker.wait(lk,[&]{
                        return (next_task_idx.load() < total_tasks.load()) || done.load();
                    });
                    if(done.load()) return;
                }
                int t_total = total_tasks.load();
                int i;
                while ((i = next_task_idx.fetch_add(1)) < t_total) {
                    current_runnable->runTask(i, t_total);
                    
                    if (completed_tasks.fetch_add(1) + 1 == t_total) {
                        std::lock_guard<std::mutex> lk(mtx);
                        cv_main.notify_all();
                    }
                }
            }
        }));
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    done.store(true);
    cv_worker.notify_all();
    for(int i = 0; i < num_threads-1; i++) thread_pool[i].join();
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    current_runnable = runnable;
    next_task_idx.store(0);
    completed_tasks.store(0);
    total_tasks.store(num_total_tasks);

    cv_worker.notify_all();
    int local_task;
    while((local_task = next_task_idx.fetch_add(1)) < num_total_tasks)
    {
        runnable->runTask(local_task, num_total_tasks);
        completed_tasks.fetch_add(1);
    }

    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_main.wait(lk, [&] {
            return completed_tasks.load() >= num_total_tasks;
        });
    }

    total_tasks.store(0);
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
