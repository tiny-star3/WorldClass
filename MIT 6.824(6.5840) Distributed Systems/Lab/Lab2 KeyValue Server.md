# Lab 2: Key/Value Server

**实验原址**：[6.5840 Lab 2: Key/Value Server](https://pdos.csail.mit.edu/6.824/labs/lab-kvsrv1.html)  
非常感谢老师的付出和开源，以下是作业介绍和我的实现(特别感谢 Google AI Studio 提供远程指导😝)   

## Key/value server with reliable network

```go
package kvsrv

import (
	"6.5840/kvsrv1/rpc"
	kvtest "6.5840/kvtest1"
	tester "6.5840/tester1"
)

type Clerk struct {
	clnt   *tester.Clnt
	server string
}

func MakeClerk(clnt *tester.Clnt, server string) kvtest.IKVClerk {
	ck := &Clerk{clnt: clnt, server: server}
	// You may add code here.
	return ck
}

// Get fetches the current value and version for a key.  It returns
// ErrNoKey if the key does not exist. It keeps trying forever in the
// face of all other errors.
//
// You can send an RPC with code like this:
// ok := ck.clnt.Call(ck.server, "KVServer.Get", &args, &reply)
//
// The types of args and reply (including whether they are pointers)
// must match the declared types of the RPC handler function's
// arguments. Additionally, reply must be passed as a pointer.
func (ck *Clerk) Get(key string) (string, rpc.Tversion, rpc.Err) {
	// You will have to modify this function.
	var args rpc.GetArgs
	args.Key = key
	var reply rpc.GetReply
	for {
		ok := ck.clnt.Call(ck.server, "KVServer.Get", &args, &reply)
		if !ok {
			continue
		}
		if reply.Err == rpc.ErrNoKey {
			break
		} else if reply.Err == rpc.OK {
			return reply.Value, reply.Version, reply.Err
		}
	}
	return "", 0, rpc.ErrNoKey
}

// Put updates key with value only if the version in the
// request matches the version of the key at the server.  If the
// versions numbers don't match, the server should return
// ErrVersion.  If Put receives an ErrVersion on its first RPC, Put
// should return ErrVersion, since the Put was definitely not
// performed at the server. If the server returns ErrVersion on a
// resend RPC, then Put must return ErrMaybe to the application, since
// its earlier RPC might have been processed by the server successfully
// but the response was lost, and the Clerk doesn't know if
// the Put was performed or not.
//
// You can send an RPC with code like this:
// ok := ck.clnt.Call(ck.server, "KVServer.Put", &args, &reply)
//
// The types of args and reply (including whether they are pointers)
// must match the declared types of the RPC handler function's
// arguments. Additionally, reply must be passed as a pointer.
func (ck *Clerk) Put(key, value string, version rpc.Tversion) rpc.Err {
	// You will have to modify this function.
	args := rpc.PutArgs{
		Key:     key,
		Value:   value,
		Version: version,
	}
	first := true
	for {
		var reply rpc.PutReply
		ok := ck.clnt.Call(ck.server, "KVServer.Put", &args, &reply)
		if !ok {
			first = false
			// 屏蔽网络错误, 无限重试直到服务器活过来并给一个明确的 reply
			continue
		}

		if reply.Err == rpc.OK {
			return rpc.OK
		}
		if reply.Err == rpc.ErrVersion {
			if first {
				return rpc.ErrVersion
			} else {
				return rpc.ErrMaybe
			}
		}
		return reply.Err
	}
}
```

```go
package kvsrv

import (
	"log"
	"sync"

	"6.5840/kvsrv1/rpc"
	"6.5840/labrpc"
	tester "6.5840/tester1"
)

const Debug = false

func DPrintf(format string, a ...interface{}) (n int, err error) {
	if Debug {
		log.Printf(format, a...)
	}
	return
}

type Value struct {
	Value   string
	Version rpc.Tversion
}

type KVServer struct {
	mu    sync.Mutex
	kvMap map[string]Value
	// Your definitions here.
}

func MakeKVServer() *KVServer {
	kv := &KVServer{}
	// Your code here.
	kv.kvMap = make(map[string]Value)
	return kv
}

// Get returns the value and version for args.Key, if args.Key
// exists. Otherwise, Get returns ErrNoKey.
func (kv *KVServer) Get(args *rpc.GetArgs, reply *rpc.GetReply) {
	// Your code here.
	kv.mu.Lock()
	defer kv.mu.Unlock()
	value, ok := kv.kvMap[args.Key]
	if ok {
		reply.Value = value.Value
		reply.Version = value.Version
		reply.Err = rpc.OK
	} else {
		reply.Err = rpc.ErrNoKey
	}
}

// Update the value for a key if args.Version matches the version of
// the key on the server. If versions don't match, return ErrVersion.
// If the key doesn't exist, Put installs the value if the
// args.Version is 0, and returns ErrNoKey otherwise.
func (kv *KVServer) Put(args *rpc.PutArgs, reply *rpc.PutReply) {
	// Your code here.
	kv.mu.Lock()
	defer kv.mu.Unlock()
	value, ok := kv.kvMap[args.Key]
	if ok {
		if value.Version == args.Version {
			kv.kvMap[args.Key] = Value{args.Value, args.Version + 1}
			reply.Err = rpc.OK
		} else {
			reply.Err = rpc.ErrVersion
		}
	} else {
		if args.Version == 0 {
			kv.kvMap[args.Key] = Value{args.Value, 1}
			reply.Err = rpc.OK
		} else {
			reply.Err = rpc.ErrNoKey
		}
	}
}

// You can ignore all arguments; they are for replicated KVservers
func StartKVServer(tc *tester.TesterClnt, ends []*labrpc.ClientEnd, gid tester.Tgid, srv int, persister *tester.Persister) []any {
	kv := MakeKVServer()
	return []any{kv}
}
```

```bash
$ make RUN="-run Reliable" kvsrv1
go build -race -o main/kvsrv1d main/kvsrv1d.go
cd kvsrv1 && go test -v -race -run Reliable
=== RUN   TestReliablePut
One client and reliable Put (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs     5 #Ops    5
--- PASS: TestReliablePut (0.13s)
=== RUN   TestPutConcurrentReliable
Test: many clients racing to put values to the same key (reliable network)...
  ... Passed --  time  1.7s #peers 1 #RPCs  3145 #Ops 6290
--- PASS: TestPutConcurrentReliable (1.84s)
=== RUN   TestMemPutManyClientsReliable
Test: memory use many put clients (reliable network)...
  ... Passed --  time 25.1s #peers 1 #RPCs 20000 #Ops 20000
--- PASS: TestMemPutManyClientsReliable (47.13s)
PASS
ok      6.5840/kvsrv1   50.164s
```

## Implementing a lock using key/value clerk

```go
package lock

import (
	"6.5840/kvsrv1/rpc"
	kvtest "6.5840/kvtest1"
)

type Lock struct {
	// IKVClerk is a go interface for k/v clerks: the interface hides
	// the specific Clerk type of ck but promises that ck supports
	// Put and Get.  The tester passes the clerk in when calling
	// MakeLock().
	ck kvtest.IKVClerk
	// You may add code here
	lockname string
	value    string
}

// The tester calls MakeLock() and passes in a k/v clerk; your code can
// perform a Put or Get by calling lk.ck.Put() or lk.ck.Get().
//
// This interface supports multiple locks by means of the
// lockname argument; locks with different names should be
// independent.
func MakeLock(ck kvtest.IKVClerk, lockname string) *Lock {
	lk := &Lock{ck: ck}
	// You may add code here
	lk.lockname = lockname
	lk.value = kvtest.RandValue(8)
	for {
		_, _, err := lk.ck.Get(lk.lockname)
		// 当前锁未创建
		if err == rpc.ErrNoKey {
			err = lk.ck.Put(lockname, "", 0)
			if err == rpc.OK {
				break
			}
		} else {
			break
		}
	}
	return lk
}

func (lk *Lock) Acquire() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		if err == rpc.ErrNoKey {
			return
		}
		// 空字符串，当前无人持锁
		if len(value) == 0 {
			err = lk.ck.Put(lk.lockname, lk.value, version)
			if err == rpc.OK {
				return
			}
		}
	}
}

func (lk *Lock) Release() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		if err == rpc.ErrNoKey {
			return
		}
		if value == lk.value {
			err = lk.ck.Put(lk.lockname, "", version)
			if err == rpc.OK {
				return
			}
		} else {
			// 未知故障
			return
		}
	}
}
```

```bash
$ make RUN="-run Reliable" lock1
go build -race -o main/kvsrv1d main/kvsrv1d.go
cd kvsrv1/lock; go test -v -race -run Reliable
=== RUN   TestReliableBasic
Test: a single Acquire and Release (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs     6 #Ops    6
--- PASS: TestReliableBasic (0.12s)
=== RUN   TestReliableNested
Test: one client, two locks (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs    24 #Ops   24
--- PASS: TestReliableNested (0.14s)
=== RUN   TestOneClientReliable
Test: 1 lock clients (reliable network)...
  ... Passed --  time  2.0s #peers 1 #RPCs   790 #Ops  790
--- PASS: TestOneClientReliable (2.12s)
=== RUN   TestManyClientsReliable
Test: 10 lock clients (reliable network)...
  ... Passed --  time  2.2s #peers 1 #RPCs  8637 #Ops 8637
--- PASS: TestManyClientsReliable (2.31s)
PASS
ok      6.5840/kvsrv1/lock      5.701s
```

**Google AI Studio**  
	这个版本更加健壮，能够正确处理分布式环境下的不确定性：

```go
func (lk *Lock) Acquire() {
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		
		// 场景1：锁已经是我的了（可能上次 Put 成功了但返回 ErrMaybe），每个客户端都要一份lk，里面的lk.value值不同
		if err == rpc.OK && value == lk.value {
			return
		}

		// 场景2：锁是空的，尝试抢锁
		if err == rpc.ErrNoKey || (err == rpc.OK && len(value) == 0) {
			// 如果 err 是 ErrNoKey，我们尝试用 version 0 写入
			v := version
			if err == rpc.ErrNoKey {
				v = 0
			}
			putErr := lk.ck.Put(lk.lockname, lk.value, v)
			if putErr == rpc.OK {
				return
			}
			// 如果是 ErrMaybe，下次循环 Get 会走“场景1”确认是否成功
		}
		
		// 场景3：锁被别人占着，或者 Put 失败，继续轮询 (Spin)
	}
}

func (lk *Lock) Release() {
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		
		// 如果 key 不存在，或者已经是空的了，说明锁已经释放
		if err == rpc.ErrNoKey || (err == rpc.OK && len(value) == 0) {
			return
		}

		// 如果锁是我持有的，尝试释放
		if value == lk.value {
			putErr := lk.ck.Put(lk.lockname, "", version)
			if putErr == rpc.OK {
				return
			}
			// 如果是 ErrMaybe，下次循环 Get 会确认是否已经变成了空
		} else {
			// 锁被别人抢走了，或者发生了异常，但在简单的锁逻辑中
			// 既然值不是我的，我没法释放，直接退出
			return
		}
	}
}
```

## Key/value server with dropped messages

在 Key/value server with reliable network 基础上加入 Before the client retries, it should wait a little bit; you can use go's `time` package and call `time.Sleep(100 * time.Millisecond)`

```go
package kvsrv

import (
	"time"

	"6.5840/kvsrv1/rpc"
	kvtest "6.5840/kvtest1"
	tester "6.5840/tester1"
)

type Clerk struct {
	clnt   *tester.Clnt
	server string
}

func MakeClerk(clnt *tester.Clnt, server string) kvtest.IKVClerk {
	ck := &Clerk{clnt: clnt, server: server}
	// You may add code here.
	return ck
}

// Get fetches the current value and version for a key.  It returns
// ErrNoKey if the key does not exist. It keeps trying forever in the
// face of all other errors.
//
// You can send an RPC with code like this:
// ok := ck.clnt.Call(ck.server, "KVServer.Get", &args, &reply)
//
// The types of args and reply (including whether they are pointers)
// must match the declared types of the RPC handler function's
// arguments. Additionally, reply must be passed as a pointer.
func (ck *Clerk) Get(key string) (string, rpc.Tversion, rpc.Err) {
	// You will have to modify this function.
	var args rpc.GetArgs
	args.Key = key
	var reply rpc.GetReply
	for {
		ok := ck.clnt.Call(ck.server, "KVServer.Get", &args, &reply)
		if !ok {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		if reply.Err == rpc.ErrNoKey {
			break
		} else if reply.Err == rpc.OK {
			return reply.Value, reply.Version, reply.Err
		}
	}
	return "", 0, rpc.ErrNoKey
}

// Put updates key with value only if the version in the
// request matches the version of the key at the server.  If the
// versions numbers don't match, the server should return
// ErrVersion.  If Put receives an ErrVersion on its first RPC, Put
// should return ErrVersion, since the Put was definitely not
// performed at the server. If the server returns ErrVersion on a
// resend RPC, then Put must return ErrMaybe to the application, since
// its earlier RPC might have been processed by the server successfully
// but the response was lost, and the Clerk doesn't know if
// the Put was performed or not.
//
// You can send an RPC with code like this:
// ok := ck.clnt.Call(ck.server, "KVServer.Put", &args, &reply)
//
// The types of args and reply (including whether they are pointers)
// must match the declared types of the RPC handler function's
// arguments. Additionally, reply must be passed as a pointer.
func (ck *Clerk) Put(key, value string, version rpc.Tversion) rpc.Err {
	// You will have to modify this function.
	args := rpc.PutArgs{
		Key:     key,
		Value:   value,
		Version: version,
	}
	first := true
	for {
		var reply rpc.PutReply
		ok := ck.clnt.Call(ck.server, "KVServer.Put", &args, &reply)
		if !ok {
			first = false
			// 屏蔽网络错误, 无限重试直到服务器活过来并给一个明确的 reply
			time.Sleep(100 * time.Millisecond)
			continue
		}

		if reply.Err == rpc.OK {
			return rpc.OK
		}
		if reply.Err == rpc.ErrVersion {
			if first {
				return rpc.ErrVersion
			} else {
				return rpc.ErrMaybe
			}
		}
		return reply.Err
	}
}
```

```bash
$ make kvsrv1
go build -race -o main/kvsrv1d main/kvsrv1d.go
cd kvsrv1 && go test -v -race
=== RUN   TestReliablePut
One client and reliable Put (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs     5 #Ops    5
--- PASS: TestReliablePut (0.12s)
=== RUN   TestPutConcurrentReliable
Test: many clients racing to put values to the same key (reliable network)...
  ... Passed --  time  2.2s #peers 1 #RPCs  3597 #Ops 7194
--- PASS: TestPutConcurrentReliable (2.30s)
=== RUN   TestMemPutManyClientsReliable
Test: memory use many put clients (reliable network)...
  ... Passed --  time 24.5s #peers 1 #RPCs 20000 #Ops 20000
--- PASS: TestMemPutManyClientsReliable (45.78s)
=== RUN   TestUnreliableNet
One client (unreliable network)...
  ... Passed --  time  9.8s #peers 1 #RPCs   278 #Ops  448
--- PASS: TestUnreliableNet (9.94s)
PASS
ok      6.5840/kvsrv1   59.195s
```

## Implementing a lock using key/value clerk and unreliable network

```go
package lock

import (
	"6.5840/kvsrv1/rpc"
	kvtest "6.5840/kvtest1"
)

type Lock struct {
	// IKVClerk is a go interface for k/v clerks: the interface hides
	// the specific Clerk type of ck but promises that ck supports
	// Put and Get.  The tester passes the clerk in when calling
	// MakeLock().
	ck kvtest.IKVClerk
	// You may add code here
	lockname string
	value    string
}

// The tester calls MakeLock() and passes in a k/v clerk; your code can
// perform a Put or Get by calling lk.ck.Put() or lk.ck.Get().
//
// This interface supports multiple locks by means of the
// lockname argument; locks with different names should be
// independent.
func MakeLock(ck kvtest.IKVClerk, lockname string) *Lock {
	lk := &Lock{ck: ck}
	// You may add code here
	lk.lockname = lockname
	lk.value = kvtest.RandValue(8)
	for {
		_, _, err := lk.ck.Get(lk.lockname)
		// 当前锁未创建
		if err == rpc.ErrNoKey {
			err = lk.ck.Put(lockname, "", 0)
			if err == rpc.OK {
				break
			}
		} else {
			break
		}
	}
	return lk
}

func (lk *Lock) Acquire() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		if err == rpc.ErrNoKey {
			return
		}
		// 如果已经有人持锁，查看是否是自己的，是自己的就退出，可能是重复发送，之前成功了
		// 每个客户端都要一份lk，里面的lk.value值不同
		if err == rpc.OK && value == lk.value {
			return
		}
		// 空字符串，当前无人持锁
		if len(value) == 0 {
			err = lk.ck.Put(lk.lockname, lk.value, version)
			if err == rpc.OK {
				return
			}
		}
	}
}

func (lk *Lock) Release() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		if err == rpc.ErrNoKey {
			return
		}
		// 如果已经锁已经释放，就退出，可能是重复发送，之前成功了
		if err == rpc.OK && len(value) == 0 {
			return
		}
		if value == lk.value {
			err = lk.ck.Put(lk.lockname, "", version)
			if err == rpc.OK {
				return
			}
		} else {
			// 未知故障
			return
		}
	}
}
```

```bash
$ make lock1
go build -race -o main/kvsrv1d main/kvsrv1d.go
cd kvsrv1/lock; go test -v -race
=== RUN   TestReliableBasic
Test: a single Acquire and Release (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs     6 #Ops    6
--- PASS: TestReliableBasic (0.12s)
=== RUN   TestReliableNested
Test: one client, two locks (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs    24 #Ops   24
--- PASS: TestReliableNested (0.14s)
=== RUN   TestOneClientReliable
Test: 1 lock clients (reliable network)...
  ... Passed --  time  2.0s #peers 1 #RPCs   797 #Ops  797
--- PASS: TestOneClientReliable (2.13s)
=== RUN   TestManyClientsReliable
Test: 10 lock clients (reliable network)...
  ... Passed --  time  2.2s #peers 1 #RPCs  8052 #Ops 8052
--- PASS: TestManyClientsReliable (2.33s)
=== RUN   TestOneClientUnreliable
Test: 1 lock clients (unreliable network)...
  ... Passed --  time  2.1s #peers 1 #RPCs    54 #Ops   42
--- PASS: TestOneClientUnreliable (2.23s)
=== RUN   TestManyClientsUnreliable
Test: 10 lock clients (unreliable network)...
  ... Passed --  time  4.7s #peers 1 #RPCs  1025 #Ops  835
--- PASS: TestManyClientsUnreliable (4.80s)
PASS
ok      6.5840/kvsrv1/lock      12.765s
```

**Google AI Studio修改建议**  

```go
package lock

import (
	"time"

	"6.5840/kvsrv1/rpc"
	kvtest "6.5840/kvtest1"
)

type Lock struct {
	// IKVClerk is a go interface for k/v clerks: the interface hides
	// the specific Clerk type of ck but promises that ck supports
	// Put and Get.  The tester passes the clerk in when calling
	// MakeLock().
	ck kvtest.IKVClerk
	// You may add code here
	lockname string
	value    string
}

// The tester calls MakeLock() and passes in a k/v clerk; your code can
// perform a Put or Get by calling lk.ck.Put() or lk.ck.Get().
//
// This interface supports multiple locks by means of the
// lockname argument; locks with different names should be
// independent.
func MakeLock(ck kvtest.IKVClerk, lockname string) *Lock {
	lk := &Lock{ck: ck}
	// You may add code here
	lk.lockname = lockname
	lk.value = kvtest.RandValue(8)
	for {
		_, _, err := lk.ck.Get(lk.lockname)
		// 当前锁未创建
		if err == rpc.ErrNoKey {
			err = lk.ck.Put(lockname, "", 0)
			if err == rpc.OK {
				break
			}
		} else {
			break
		}
	}
	return lk
}

func (lk *Lock) Acquire() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		// 如果因为某种原因（比如服务器重启、数据丢失或者 MakeLock 初始化还没完成）
		// 尝试新建一个锁
		if err == rpc.ErrNoKey {
			err = lk.ck.Put(lk.lockname, lk.value, 0)
			if err == rpc.OK {
				return
			}
		}
		// 如果已经有人持锁，查看是否是自己的，是自己的就退出，可能是重复发送，之前成功了
		// 每个客户端都要一份lk，里面的lk.value值不同
		if err == rpc.OK && value == lk.value {
			return
		}
		// 空字符串，当前无人持锁
		if len(value) == 0 {
			err = lk.ck.Put(lk.lockname, lk.value, version)
			if err == rpc.OK {
				return
			}
		}
		// 加入退避（Backoff）, 增加休眠，对网络更友好
		time.Sleep(100 * time.Millisecond)
	}
}

func (lk *Lock) Release() {
	// Your code here
	for {
		value, version, err := lk.ck.Get(lk.lockname)
		if err == rpc.ErrNoKey {
			return
		}
		// 如果已经锁已经释放，就退出，可能是重复发送，之前成功了
		if err == rpc.OK && len(value) == 0 {
			return
		}
		if value == lk.value {
			err = lk.ck.Put(lk.lockname, "", version)
			if err == rpc.OK {
				return
			}
		} else {
			// 未知故障
			return
		}
	}
}
```

```bash
$ make lock1
go build -race -o main/kvsrv1d main/kvsrv1d.go
cd kvsrv1/lock; go test -v -race
=== RUN   TestReliableBasic
Test: a single Acquire and Release (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs     6 #Ops    6
--- PASS: TestReliableBasic (0.12s)
=== RUN   TestReliableNested
Test: one client, two locks (reliable network)...
  ... Passed --  time  0.0s #peers 1 #RPCs    24 #Ops   24
--- PASS: TestReliableNested (0.14s)
=== RUN   TestOneClientReliable
Test: 1 lock clients (reliable network)...
  ... Passed --  time  2.0s #peers 1 #RPCs   783 #Ops  783
--- PASS: TestOneClientReliable (2.13s)
=== RUN   TestManyClientsReliable
Test: 10 lock clients (reliable network)...
  ... Passed --  time  2.7s #peers 1 #RPCs  1176 #Ops 1176
--- PASS: TestManyClientsReliable (2.81s)
=== RUN   TestOneClientUnreliable
Test: 1 lock clients (unreliable network)...
  ... Passed --  time  2.2s #peers 1 #RPCs    68 #Ops   56
--- PASS: TestOneClientUnreliable (2.27s)
=== RUN   TestManyClientsUnreliable
Test: 10 lock clients (unreliable network)...
  ... Passed --  time  4.7s #peers 1 #RPCs   456 #Ops  366
--- PASS: TestManyClientsUnreliable (4.86s)
PASS
ok      6.5840/kvsrv1/lock      13.364s
```

