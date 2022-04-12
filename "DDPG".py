"DDPG"

def DDPG(t_steps, num_episodes, print_every):
    
    buffer_size = int(1e5)
    batch_size = 256
    agent = Agent(batch_size, buffer_size, state_size=33, action_size=4, hidden_layer=300, hidden2_layer=400)
    recent_scores = deque(maxlen=100)
    scores_list = np.zeros(1)
    scores = []
    
    for x in range(num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        for t in range(t_steps):
            states = env_info.vector_observations
            actions = agent.act(states)
            policy = env.step(actions)[brain_name]
            rewards = policy.rewards
            dones = policy.local_done
            next_states = policy.vector_observations
            update = agent.step(states, actions, rewards, next_states, dones)
            scores_list += rewards
            if np.any(dones):
                break
        current_mean = np.mean(scores_list)
        recent_scores.append(current_mean)
        scores.append(current_mean)
        
        
        print_every_ = num_episodes % print_every
            
        if print_every_ == 0:
            print("Episode number: {}, Total Score Mean: {}".format(x, current_mean))
            print("Recent Score Mean: {}".format(np.mean(recent_scores)))
            
    return scores, agent, recent_scores