#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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
    this->num_threads = num_threads;
    for(int i = 0; i < num_threads; i++)
    {
        thread_pool.emplace_back(std::thread([this](){
            while (true) {
                std::shared_ptr<task> curr = nullptr;
                int i = 0;
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    // 当可运行队列非空或者通知结束的时候，唤醒线程
                    cv_worker.wait(lk,[&]{
                        return !run_queue.empty() || done.load();
                    });
                    // 通知结束时，线程退出
                    if(done.load()) return;
                    // std::cout<< std::this_thread::get_id() <<std::endl;
                    curr = run_queue.front();
                    // 当当前线程拿的是当前批次最后一个任务时，pop可运行队列
                    if((i = curr->next_task_idx.fetch_add(1)) == curr->total_tasks-1){
                        run_queue.pop();
                    }
                    if(i >= curr->total_tasks) continue; 
                }
                int total_tasks = curr->total_tasks;
                // std::cout<< i << " " << curr->total_tasks << " " << curr->task_id << " " << std::this_thread::get_id() << std::endl<<std::endl;
                curr->runnable->runTask(i, total_tasks);
                
                // 当当前线程完成的是目前批次最后一个任务时，处理依赖关系
                if(curr->completed_tasks.fetch_add(1) + 1 == total_tasks) {
                    std::unique_lock<std::mutex> lk(mtx);
                    // std::cout<< i << " " << curr->completed_tasks << " " << curr->total_tasks << " " << curr->task_id << " " << std::this_thread::get_id() << std::endl<<std::endl;
                    id_finished.insert(curr->task_id);
                    // std::cout << "id_finished: " << curr->task_id << std::endl;
                    auto it = deps.find(curr->task_id);
                    if(it != deps.end())
                    {
                        for(auto dep:it->second)
                        {
                            if(--deps_total[dep] == 0)
                            {
                                run_queue.push(std::move(id_task[dep]));
                                deps_total.erase(dep);
                                id_task.erase(dep);
                            }
                        }
                        deps.erase(it);
                    }
                    // 当当前完成批次任务等于总批次任务时，唤醒主线程，即sync
                    if((int)id_finished.size() == total_task_id)
                    {
                        // std::cout<<"aaa"<<std::endl;
                        lk.unlock();
                        cv_main.notify_all();
                        lk.lock();
                    }
                    lk.unlock();
                    cv_worker.notify_all();
                    lk.lock();
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
    // 修改done结束标志，停止线程池运行
    done.store(true);
    // 通知工作线程结束
    cv_worker.notify_all();
    for(int i = 0; i < num_threads; i++) thread_pool[i].join();
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    runAsyncWithDeps(runnable, num_total_tasks, std::vector<TaskID>());
    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //
    std::unique_lock<std::mutex> lk(mtx);
    // std::cout << "now_task_id: " << total_task_id << std::endl;
    // 如果没有依赖关系，直接加入可以执行队列
    if(deps.empty())
    {
        run_queue.push(std::make_shared<task>(runnable, total_task_id, num_total_tasks));
    }else{
        // 记录依赖
        for(auto i:deps)
        {
            // 若当前任务已经完成，则不记录依赖
            if(id_finished.find(i) == id_finished.end()){
                // std::cout << "deps: " << i << " " << total_task_id << std::endl;
                this->deps[i].push_back(total_task_id);
                deps_total[total_task_id]++;
            }
        }
        // 若存在依赖，加入目前不能运行的任务，否则加入可以运行的队列
        if(deps_total.find(total_task_id) != deps_total.end())
            id_task[total_task_id] = std::make_shared<task>(runnable, total_task_id, num_total_tasks);
        else run_queue.push(std::make_shared<task>(runnable, total_task_id, num_total_tasks));
    }
    int curr_task_id = total_task_id++;
    lk.unlock();
    // 通知工作线程运行
    cv_worker.notify_all();
    lk.lock();
    return curr_task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    {
        std::unique_lock<std::mutex> lk(mtx);
        // 等待所有批次任务完成被唤醒
        cv_main.wait(lk, [&] {
            return total_task_id == (int)id_finished.size();
        });
    }
    return;
}
