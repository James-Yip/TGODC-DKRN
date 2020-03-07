from target_chat import target_chat_instance

num_sessions = input('Please input the number of sessions you wanna try: ')
for i in range(int(num_sessions)):
    responses = []
    target_chat_instance.chat()
    while len(responses) < 2:
        responses = target_chat_instance.chat(input('HUMAN: '))
