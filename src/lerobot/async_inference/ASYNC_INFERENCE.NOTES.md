# Robot Client
## FIFO Queue Incorrect Data Structure
The queue is introspected via `Queue.queue` because we need to sort by actions by logical timestamps. If you need to introspect and mutate the queue then you should probably not be using a queue. Regular usuage of the queue is threadsafe but once you access and mutate the internal queue representation it's no longer threadsafe, this is why the external action_queue_lock lock needs to be used which is more of a bandaid than a proper solution.

A SortedDict keyed on logical timestamp would probably be a better data structure https://grantjenks.com/docs/sortedcontainers/sorteddict.html

## Server
### Actions/Endpoints
- Ready - Called in robot client start `self.stub.Ready(services_pb2.Empty())`
- SendPolicyInstructions  - Called in robot client start `self.stub.SendPolicyInstructions(policy_setup)`
- SendObservations - Called in robot client control_loop_observation `self.send_observation(observation)`
- GetActions - Called in robot client receive_actions `actions_chunk = self.stub.GetActions(services_pb2.Empty())`

## Client
- async_client - Main entry point for the robot client
- client.start() - Calls/Sends Ready and SendPolicyInstructions to the policy server
    - Starts monitor_queue_level in another background thread.
    - monitor_queue_level acquires action_queue_lock when reading from action_queue
    - self.must_go.set() is conditionally set. Is defined as `self.must_go = threading.Event()` in the construct, is thread safe and is not set in context of lock.
- run client.receive_actions in a background thread. Has barrier(2).wait(). Waiting for this thread and the main thread
- start the control loop in the main thread. Has barrier(2).wait(). Waiting for this thread (main) and the receive_actions background thread