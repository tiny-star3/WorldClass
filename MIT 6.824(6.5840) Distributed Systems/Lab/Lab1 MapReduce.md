# Lab 1: MapReduce

**实验原址**：[6.5840 Lab 1: MapReduce](https://pdos.csail.mit.edu/6.824/labs/lab-mr.html)  
非常感谢老师的付出和开源，以下是作业介绍和我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## coordinator.go

```go
package mr

import (
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"sync"
	"time"
)

type Coordinator struct {
	// Your definitions here.
	// Map任务
	Files            []string      // Map任务要处理的文件名
	MapTasks         int           // 已经分配出去的文件名数
	FinishedMapNum   int           // 已经完成的Map任务(文件名数)
	FinishedMapTasks map[int]bool  // 标记已完成的任务
	MapTime          map[int]int64 // 当前任务时间戳,超出10s视为崩溃，启动恢复

	// 控制变量
	Mu sync.Mutex

	// Reduce任务
	NReduce             int
	ReduceTasks         int           // 已经分配出去的Reduce任务
	FinishedReduceNum   int           // 已经完成的Reduce任务
	FinishedReduceTasks map[int]bool  // 标记已完成的任务
	ReduceTime          map[int]int64 // 当前任务时间戳,超出10s视为崩溃，启动恢复
}

// Your code here -- RPC handlers for the worker to call.

// an example RPC handler.
//
// the RPC argument and reply types are defined in rpc.go.
func (c *Coordinator) Example(args *ExampleArgs, reply *ExampleReply) error {
	reply.Y = args.X + 1
	return nil
}

func (c *Coordinator) RequestWork(args *WorkArgs, reply *WorkReply) error {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	if args.WorkId == -1 {
		// 请求任务
		if c.FinishedMapNum < len(c.Files) && c.MapTasks < len(c.Files) {
			// 还有Map任务可分配
			reply.WorkId = 0
			reply.NReduce = c.NReduce
			reply.Filename = c.Files[c.MapTasks]
			reply.TaskNum = c.MapTasks
			c.MapTime[c.MapTasks] = time.Now().Unix()
			c.MapTasks++
		} else if c.FinishedMapNum < len(c.Files) && c.MapTasks == len(c.Files) {
			// Map任务都已经分配，但还未全部完成，等待
			// 查看是否有超时(10s)崩溃，有的话启动
			nowTime := time.Now().Unix()
			for i := 0; i < len(c.Files); i++ {
				if c.FinishedMapTasks[i] {
					continue
				}
				if nowTime-c.MapTime[i] > 10 {
					// 超时崩溃
					reply.WorkId = 0
					reply.NReduce = c.NReduce
					reply.Filename = c.Files[i]
					reply.TaskNum = i
					c.MapTime[i] = nowTime
					return nil
				}
			}
			reply.WorkId = 2
		} else if c.FinishedMapNum == len(c.Files) && c.ReduceTasks < c.NReduce {
			// Map任务都已经完成，分配Reduce任务
			reply.WorkId = 1
			reply.NReduce = c.NReduce
			reply.TaskNum = c.ReduceTasks
			c.ReduceTime[c.ReduceTasks] = time.Now().Unix()
			c.ReduceTasks++
		} else if c.FinishedMapNum == len(c.Files) && c.ReduceTasks == c.NReduce && c.FinishedReduceNum < c.NReduce {
			// Reduce任务都已经分配，但还未全部完成，等待
			// 查看是否有超时(10s)崩溃，有的话启动
			nowTime := time.Now().Unix()
			for i := 0; i < c.NReduce; i++ {
				if c.FinishedReduceTasks[i] {
					continue
				}
				if nowTime-c.ReduceTime[i] > 10 {
					// 超时崩溃
					reply.WorkId = 1
					reply.NReduce = c.NReduce
					reply.TaskNum = i
					c.ReduceTime[i] = nowTime
					return nil
				}
			}
			reply.WorkId = 2
		} else if c.FinishedReduceNum == c.NReduce {
			// 所有任务都已经完成
			reply.WorkId = 3
		}
	} else {
		// fmt.Println(args.WorkId, "Work Done2")
		// 任务完成
		if c.FinishedMapNum < len(c.Files) && args.TaskType == 0 {
			// Map任务完成
			if c.FinishedMapTasks[args.WorkId] == false {
				c.FinishedMapTasks[args.WorkId] = true
				c.FinishedMapNum++
			}
		} else if c.FinishedReduceNum < c.NReduce && args.TaskType == 1 {
			// Reduce任务完成
			if c.FinishedReduceTasks[args.WorkId] == false {
				c.FinishedReduceTasks[args.WorkId] = true
				c.FinishedReduceNum++
			}
			if c.FinishedReduceNum == c.NReduce {
				reply.WorkId = 3
			}
		}
	}
	return nil
}

// start a thread that listens for RPCs from worker.go
func (c *Coordinator) server(sockname string) {
	rpc.Register(c)
	rpc.HandleHTTP()
	os.Remove(sockname)
	l, e := net.Listen("unix", sockname)
	if e != nil {
		log.Fatalf("listen error %s: %v", sockname, e)
	}
	go http.Serve(l, nil)
}

// main/mrcoordinator.go calls Done() periodically to find out
// if the entire job has finished.
func (c *Coordinator) Done() bool {
	c.Mu.Lock()
	defer c.Mu.Unlock()
	return c.FinishedReduceNum == c.NReduce
}

// create a Coordinator.
// main/mrcoordinator.go calls this function.
// nReduce is the number of reduce tasks to use.
func MakeCoordinator(sockname string, files []string, nReduce int) *Coordinator {
	c := Coordinator{}

	// Your code here.
	c.Files = files
	c.MapTasks = 0
	c.FinishedMapNum = 0
	c.FinishedMapTasks = make(map[int]bool)
	c.FinishedReduceNum = 0
	c.FinishedReduceTasks = make(map[int]bool)
	c.ReduceTasks = 0
	c.MapTime = make(map[int]int64)
	c.ReduceTime = make(map[int]int64)
	c.NReduce = nReduce
	c.server(sockname)
	return &c
}
```

## worker.go

```go
package mr

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/rpc"
	"os"
	"regexp"
	"sort"
	"strconv"
	"time"
)

// Map functions return a slice of KeyValue.
type KeyValue struct {
	Key   string
	Value string
}

// for sorting by key.
type ByKey []KeyValue

// for sorting by key.
func (a ByKey) Len() int           { return len(a) }
func (a ByKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByKey) Less(i, j int) bool { return a[i].Key < a[j].Key }

// use ihash(key) % NReduce to choose the reduce
// task number for each KeyValue emitted by Map.
func ihash(key string) int {
	h := fnv.New32a()
	h.Write([]byte(key))
	return int(h.Sum32() & 0x7fffffff)
}

var coordSockName string // socket for coordinator

// main/mrworker.go calls this function.
func Worker(sockname string, mapf func(string, string) []KeyValue,
	reducef func(string, []string) string) {

	coordSockName = sockname

	// Your worker implementation here.

	// uncomment to send the Example RPC to the coordinator.
	// CallExample()
	CallRequestWork(mapf, reducef)
}

func CallRequestWork(mapf func(string, string) []KeyValue, reducef func(string, []string) string) {
	for {
		args := WorkArgs{}
		args.WorkId = -1

		reply := WorkReply{}

		ok := call("Coordinator.RequestWork", &args, &reply)
		if ok {
			if reply.WorkId == 0 {
				filename := reply.Filename
				file, err := os.Open(filename)
				if err != nil {
					log.Fatalf("cannot open %v", filename)
				}
				content, err := io.ReadAll(file)
				if err != nil {
					log.Fatalf("cannot read %v", filename)
				}
				file.Close()
				kva := mapf(filename, string(content))
				buckets := [][]KeyValue{}
				for i := 0; i < reply.NReduce; i++ {
					buckets = append(buckets, []KeyValue{})
				}
				for _, kv := range kva {
					idx := ihash(kv.Key) % reply.NReduce
					buckets[idx] = append(buckets[idx], kv)
				}
				for index, bucket := range buckets {
					oname := "mr-" + strconv.Itoa(reply.TaskNum) + "-" + strconv.Itoa(index)
					tempFile, err := os.CreateTemp("", "mr-tmp-*")
					if err != nil {
						log.Fatal(err)
					}
					// 写入中间文件
					enc := json.NewEncoder(tempFile)
					for _, kv := range bucket {
						err := enc.Encode(&kv)
						if err != nil {
							log.Fatal(err)
						}
					}
					tempFile.Close()
					os.Rename(tempFile.Name(), oname)
				}
				// fmt.Println(reply.TaskNum, "Work Done")
				// 当前Map任务完成
				Doneargs := WorkArgs{}
				Doneargs.WorkId = reply.TaskNum
				Doneargs.TaskType = 0
				ok := call("Coordinator.RequestWork", &Doneargs, nil)
				if !ok {
					fmt.Printf("Done call failed!\n")
				}
			} else if reply.WorkId == 1 {
				pattern := fmt.Sprintf(`^mr.*%d$`, reply.TaskNum)
				reg := regexp.MustCompile(pattern)
				dir, _ := os.Open(".")
				files, _ := dir.Readdir(-1)
				intermediate := []KeyValue{}
				for _, f := range files {
					if !f.IsDir() && reg.MatchString(f.Name()) {
						file, err := os.Open(f.Name())
						if err != nil {
							log.Fatalf("cannot open %v", f.Name())
						}
						// 读取中间文件
						dec := json.NewDecoder(file)
						for {
							var kv KeyValue
							if err := dec.Decode(&kv); err != nil {
								break
							}
							intermediate = append(intermediate, kv)
						}
						file.Close()
					}
				}
				sort.Sort(ByKey(intermediate))
				oname := "mr-out-" + strconv.Itoa(reply.TaskNum)
				tempFile, err := os.CreateTemp("", "mr-tmp-*")
				if err != nil {
					log.Fatal(err)
				}
				i := 0
				for i < len(intermediate) {
					j := i + 1
					for j < len(intermediate) && intermediate[j].Key == intermediate[i].Key {
						j++
					}
					values := []string{}
					for k := i; k < j; k++ {
						values = append(values, intermediate[k].Value)
					}
					output := reducef(intermediate[i].Key, values)
					fmt.Fprintf(tempFile, "%v %v\n", intermediate[i].Key, output)
					i = j
				}
				tempFile.Close()
				os.Rename(tempFile.Name(), oname)
				// 当前Reduce任务完成
				Doneargs := WorkArgs{}
				Doneargs.WorkId = reply.TaskNum
				Doneargs.TaskType = 1
				ok := call("Coordinator.RequestWork", &Doneargs, nil)
				if !ok {
					fmt.Printf("Done call failed!\n")
				}
			} else if reply.WorkId == 2 {
				// 等待/请稍后再试
				time.Sleep(time.Second)
			} else if reply.WorkId == 3 {
				// 所有任务结束
				return
			}
		} else {
			fmt.Printf("Reuqest call failed!\n")
		}
	}

}

// example function to show how to make an RPC call to the coordinator.
//
// the RPC argument and reply types are defined in rpc.go.
func CallExample() {

	// declare an argument structure.
	args := ExampleArgs{}

	// fill in the argument(s).
	args.X = 99

	// declare a reply structure.
	reply := ExampleReply{}

	// send the RPC request, wait for the reply.
	// the "Coordinator.Example" tells the
	// receiving server that we'd like to call
	// the Example() method of struct Coordinator.
	ok := call("Coordinator.Example", &args, &reply)
	if ok {
		// reply.Y should be 100.
		fmt.Printf("reply.Y %v\n", reply.Y)
	} else {
		fmt.Printf("call failed!\n")
	}
}

// send an RPC request to the coordinator, wait for the response.
// usually returns true.
// returns false if something goes wrong.
func call(rpcname string, args interface{}, reply interface{}) bool {
	// c, err := rpc.DialHTTP("tcp", "127.0.0.1"+":1234")
	c, err := rpc.DialHTTP("unix", coordSockName)
	if err != nil {
		log.Fatal("dialing:", err)
	}
	defer c.Close()

	if err := c.Call(rpcname, args, reply); err == nil {
		return true
	}
	log.Printf("%d: call failed err %v", os.Getpid(), err)
	return false
}
```

## rpc.go

```go
package mr

//
// RPC definitions.
//
// remember to capitalize all names.
//

//
// example to show how to declare the arguments
// and reply for an RPC.
//

type ExampleArgs struct {
	X int
}

type ExampleReply struct {
	Y int
}

// Add your RPC definitions here.
type WorkArgs struct {
	WorkId   int // -1表示请求任务，>=0表示第n-1个任务完成
	TaskType int // 0: Map, 1: Reduce
}

type WorkReply struct {
	WorkId   int    // 0表示Map，1表示Reduce, 2表示等待/请稍后再试，3表示所有任务结束
	Filename string // 任务Map时代表文件名
	NReduce  int    // 分成的 the number of reduce tasks
	TaskNum  int    // Map/Reduce task number
}
```

## 运行结果

```bash
$ make mr
go build -race -o main/mrsequential main/mrsequential.go
go build -race -o main/mrcoordinator main/mrcoordinator.go
go build -race -o main/mrworker main/mrworker.go&
(cd mrapps && go build -race -buildmode=plugin wc.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin indexer.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin mtiming.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin rtiming.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin jobcount.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin early_exit.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin crash.go) || exit 1
(cd mrapps && go build -race -buildmode=plugin nocrash.go) || exit 1
cd mr; go test -v -race
=== RUN   TestWc
--- PASS: TestWc (12.18s)
=== RUN   TestIndexer
--- PASS: TestIndexer (7.01s)
=== RUN   TestMapParallel
--- PASS: TestMapParallel (8.08s)
=== RUN   TestReduceParallel
--- PASS: TestReduceParallel (10.20s)
=== RUN   TestJobCount
--- PASS: TestJobCount (12.21s)
=== RUN   TestEarlyExit
--- PASS: TestEarlyExit (8.10s)
=== RUN   TestCrashWorker
--- PASS: TestCrashWorker (37.89s)
PASS
ok      6.5840/mr       96.691s
```

## 优化建议

**优化 Reduce 查找中间文件的效率**  
	现在的做法是遍历整个当前目录并用正则表达式匹配  
	既然我们知道中间文件的命名规则是 mr-X-Y（其中 Y 是当前的 TaskNum，X 是 Map 的序号），我们可以直接根据 Map 的总数（len(c.Files)）拼接文件名并读取  

**超时检查的“被动”变“主动”**  
	目前是在 RequestWork 被调用时“顺便”检查超时。如果此时没有 Worker 来请求任务，即使有的任务已经超时了，Coordinator 也不会发现  
	在 MakeCoordinator 里启动一个后台协程，每秒钟扫描一次任务列表  
	这样职责更清晰：RequestWork 只负责分发任务，tick 负责监控健康