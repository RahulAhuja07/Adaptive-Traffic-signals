import os
import traci
import sumo

sumo_home = sumo.SUMO_HOME
sumo_binary = os.path.join(sumo_home, "bin", "sumo")

config = [
    sumo_binary,
    "-n", "rl_intersection.net.xml",
    "-r", "rl_routes.rou.xml",
    "--begin", "0",
    "--end", "60",
    "--no-step-log", "true",
]

traci.start(config)

try:
    for step in range(60):
        traci.simulationStep()

        if step % 10 == 0:
            vehicles = traci.vehicle.getIDList()
            traffic_lights = traci.trafficlight.getIDList()

            print(f"Step {step}: vehicles={len(vehicles)}")
            print("Traffic lights:", traffic_lights)
finally:
    traci.close()
