### Structure

Deploy use server-client structure, you should create a server on this NUC. If you want to deploy with high frequency, make sure network communication is great.

### How to create a server

You may start the server on the other computer. See `server_example.py` as reference. 
The server will recive images and language instruction from client, and it should send the action (or action sequences) to client.

### How to create a client

See `query_xxx.py`. This client will load images, robot states from franka robot. And send to the server in other computer to get actions.

By runing:

#### option 1:

You should run `ssh wuqiong3 -L 9876:localhost:9876` to start the terminal, and then keep the terminal open, then you can use the local port(9876) to link the remote server(wuqiong3:9876).

`ssh <remote_host_ssh_config> -L <local_port>:<destination_host_ip>:<destination_port>`

```bash
ssh wq3 -L 9876:localhost:9876
python deploy/query_vla_test.py --vla_server_ip "localhost" --vla_server_port 9876
# python query_vla.py --vla_server_ip "localhost" --vla_server_port 9876
```

```bash
# if you use imageio
pip install imageio[ffmpeg]
```