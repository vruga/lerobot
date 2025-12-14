# Robot Client
## FIFO Queue Incorrect Data Structure
The queue is introspected via `Queue.queue` because we need to sort by actions by logical timestamps. If you need to introspect and mutate the queue then you should probably not be using a queue. Regular usuage of the queue is threadsafe but once you access and mutate the internal queue representation it's no longer threadsafe, this is why the external action_queue_lock lock needs to be used which is more of a bandaid than a proper solution.

A SortedDict keyed on logical timestamp would probably be a better data structure https://grantjenks.com/docs/sortedcontainers/sorteddict.html

## Server
- Start server with threadpool executor with max_workers=4. This is because we want to process GetAction and SendObservations concurrently, and one GetAction or SendObservation might still be running before when the next one is called.
- observation_queue is a Queue(maxsize=1) to avoid build up. No lock is acquired because the queue semantics are not violated (not the case with action_queue in the client which reads from internal .queue).
- GetActions calls observation_queue.get with 2 second timeout. Empty observation queue is called by try catch services_pb2.Empty()

### Actions/Endpoints
- Ready - Called in robot client start `self.stub.Ready(services_pb2.Empty())`
- SendPolicyInstructions  - Called in robot client start `self.stub.SendPolicyInstructions(policy_setup)`
- SendObservations - Called in robot client control_loop_observation `self.send_observation(observation)`
- GetActions - Called in robot client receive_actions `actions_chunk = self.stub.GetActions(services_pb2.Empty())`

## Client
- async_client - Main entry point for the robot client
- client.start() - Calls/Sends Ready and SendPolicyInstructions to the policy server
    - self.must_go.set() is conditionally set. Is defined as `self.must_go = threading.Event()` in the construct, is thread safe and is not set in context of lock.
- run client.receive_actions in a background thread. Has barrier(2).wait(). Waiting for this thread and the main thread
- start the control loop in the main thread. Has barrier(2).wait(). Waiting for this thread (main) and the receive_actions background thread
- control_loop starts loop with while self.running is True.
    - control_loop_action is called if actions_available() is True. Reads action from queue, converts to action dict, sends to robot, updates latest_action.
- get_observation from the robot, send_observation to the policy server. Acquires action_queue_lock when reading from action_queue. Calls send_observation which in turn calls self.stub.SendObservations(observation_iterator)

## Alternative Client (Lock Free)
- background thread: recieve action writes calls GetAction and writes using `put` to _incoming_action_chunks queue. It is the only thread that is allowed to write to this queue, main thread is not allowed to write to this queue.
- main thread: control loop calls `get` on _incoming_action_chunks queue to get the next action chunk and then merges it into the action schedule which is a SortedDict by logical timestamp.
- main thread is now the only owner of lastest_action so no need for lock.
- main thread is now the only owner of must_go so no need for threading.Event.

## Alternative Client State Machine
Does not include GRPC handshaking for simplicity and so we can focus on the core logic.

### Alternative Client 
- main thread: starts the control loop
    - merge incoming actions into action schedule (in first tick we expect there to be no actions yet)
        - calls get on the _incoming_action_chunks queue to get the next action chunk
        - merges it into the action schedule. Sort by logical timestamp, only merge actions that are strictly newer than the latest action.
    - Run the actions on the robot with send_action and update latest_action variable
    - get observation from robot with get_observation
    - send observation to policy server with send_observation
- background thread: starts the action receiving loop
    - calls GetActions and writes using `put` to _incoming_action_chunks queue. Acts as a staging area for action chunks. The _incoming_action_chunks queue items are List[TimedAction]. This is a bounded queue to prevent backpressure when the control loop slows down. Overflow policy: drop oldest chunks and keep the newest.


## Resources
- Async inference a deep dive: https://huggingface.co/blog/async-robot-inference#async-inference-a-deep-dive