### Structure

Deploy use server-client structure, you should create a server on this NUC. If you want to deploy with high frequency, make sure network communication is great.

### How to create a server

You may start the server on the other computer. See `server_example.py` as reference. 
The server will recive images and language instruction from client, and it should send the action (or action sequences) to client.

### How to create a client

By runing:

```bash
python examples/openvla_deploy/query_server.py --vla_server_ip xx.xx.xx.xx
```