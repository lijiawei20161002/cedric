from gym.envs.registration import register

register(
    id="ddos-v0",
    entry_point="ddos_gym.envs:DDoS",
)
