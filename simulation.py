import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.special import gamma
import csv
import random
import heapq


def simulate_server_and_disks(sim_duration, shape, server):
    server_id = server['id']
    server_repair_cost = float(server['server_repair_cost_PLN'])
    server_failure_rate = float(server['failure_rate'])
    maintenance_cost_per_hour = float(server['maintenance_cost_per_hour_PLN'])
    disks = server['disks']
    num_disks = len(disks)

    server_status = 'up'
    disks_status = ['up'] * num_disks

    server_total_downtime = 0.0
    server_failures = 0
    server_total_repair_cost = 0.0
    server_total_lost_revenue = 0.0
    server_downtime_intervals = []
    server_downtime_start = None

    disks_results = []
    for disk in disks:
        disks_results.append({
            'downtime': 0.0,
            'failures': 0,
            'repair_cost_total': 0.0,
            'lost_revenue_total': 0.0,
            'mttr': 0.0,
            'mttf': 0.0,
            'availability_percent': 100.0,
            'downtime_intervals': []
        })

    total_lost_revenue_per_hour = sum(float(disk['lost_revenue_per_hour_PLN']) for disk in disks)

    events = []
    heapq.heapify(events)
    server_mttf = 1.0 / server_failure_rate if server_failure_rate > 0 else float('inf')
    server_scale = server_mttf / gamma(1 + 1 / shape)

    first_server_fail = weibull_min.rvs(shape, scale=server_scale)
    heapq.heappush(events, (first_server_fail, 'server_fail', None))

    disk_scales = []
    for disk in disks:
        disk_failure_rate = float(disk['failure_rate'])
        disk_mttf = 1.0 / disk_failure_rate if disk_failure_rate > 0 else float('inf')
        disk_scale = disk_mttf / gamma(1 + 1 / shape)
        disk_scales.append(disk_scale)

    for i in range(num_disks):
        first_disk_fail = weibull_min.rvs(shape, scale=disk_scales[i])
        heapq.heappush(events, (first_disk_fail, 'disk_fail', i))

    current_time = 0.0

    while events:
        event = heapq.heappop(events)
        event_time, event_type, component_index = event

        if event_time > sim_duration:
            break

        current_time = event_time

        if event_type == 'server_fail':
            if server_status == 'up':
                server_status = 'down'
                server_failures += 1
                server_total_repair_cost += server_repair_cost
                server_downtime_start = current_time

                repair_time = random.uniform(1.5, 2.5)
                repair_finish = current_time + repair_time
                if repair_finish > sim_duration:
                    repair_finish = sim_duration
                heapq.heappush(events, (repair_finish, 'server_repair', None))

                server_downtime_intervals.append((current_time, repair_finish))
                server_total_downtime += (repair_finish - current_time)

        elif event_type == 'server_repair':
            if server_status == 'down':
                server_status = 'up'

                if server_downtime_start is not None:
                    server_lost_revenue = (current_time - server_downtime_start) * maintenance_cost_per_hour
                    server_total_lost_revenue += server_lost_revenue

                next_server_fail = current_time + weibull_min.rvs(shape, scale=server_scale)
                if next_server_fail <= sim_duration:
                    heapq.heappush(events, (next_server_fail, 'server_fail', None))

        elif event_type == 'disk_fail':
            if server_status == 'up' and disks_status[component_index] == 'up':
                disks_status[component_index] = 'down'
                disks_results[component_index]['failures'] += 1
                disks_results[component_index]['repair_cost_total'] += float(disks[component_index]['repair_cost_PLN'])
                disks_results[component_index]['downtime_intervals'].append((current_time, None))

                repair_time = random.uniform(1.5, 2.5)
                repair_finish = current_time + repair_time
                if repair_finish > sim_duration:
                    repair_finish = sim_duration
                heapq.heappush(events, (repair_finish, 'disk_repair', component_index))

        elif event_type == 'disk_repair':
            if server_status == 'up' and disks_status[component_index] == 'down':
                disks_status[component_index] = 'up'

                if disks_results[component_index]['downtime_intervals'] and \
                        disks_results[component_index]['downtime_intervals'][-1][1] is None:
                    start, _ = disks_results[component_index]['downtime_intervals'][-1]
                    downtime = current_time - start
                    disks_results[component_index]['downtime'] += downtime

                    lost_revenue = downtime * float(disks[component_index]['lost_revenue_per_hour_PLN'])
                    disks_results[component_index]['lost_revenue_total'] += lost_revenue

                    disks_results[component_index]['downtime_intervals'][-1] = (start, current_time)

                next_disk_fail = current_time + weibull_min.rvs(shape, scale=disk_scales[component_index])
                if next_disk_fail <= sim_duration:
                    heapq.heappush(events, (next_disk_fail, 'disk_fail', component_index))

    if server_status == 'down' and server_downtime_start is not None:
        downtime = sim_duration - server_downtime_start
        server_total_downtime += downtime
        server_lost_revenue = downtime * maintenance_cost_per_hour
        server_total_lost_revenue += server_lost_revenue
        server_downtime_intervals.append((server_downtime_start, sim_duration))

    for i in range(num_disks):
        if disks_status[i] == 'down' and disks_results[i]['downtime_intervals'] and \
                disks_results[i]['downtime_intervals'][-1][1] is None:
            start, _ = disks_results[i]['downtime_intervals'][-1]
            disk_downtime = sim_duration - start
            disks_results[i]['downtime'] += disk_downtime

            lost_revenue = disk_downtime * float(disks[i]['lost_revenue_per_hour_PLN'])
            disks_results[i]['lost_revenue_total'] += lost_revenue

            disks_results[i]['downtime_intervals'][-1] = (start, sim_duration)

    for disk in disks_results:
        if disk['failures'] > 0:
            disk['mttr'] = disk['downtime'] / disk['failures']
            disk['mttf'] = sim_duration / disk['failures']
        else:
            disk['mttr'] = 0.0
            disk['mttf'] = sim_duration

        if (disk['mttf'] + disk['mttr']) > 0:
            disk['availability_percent'] = (disk['mttf'] / (disk['mttf'] + disk['mttr'])) * 100.0
        else:
            disk['availability_percent'] = 0.0

    if server_failures > 0:
        server_mttr = server_total_downtime / server_failures
        server_mttf = sim_duration / server_failures
    else:
        server_mttr = 0.0
        server_mttf = sim_duration

    if sim_duration > 0:
        server_availability_percent = (1.0 - (server_total_downtime / sim_duration)) * 100.0
    else:
        server_availability_percent = 0.0

    maintenance_cost_total = maintenance_cost_per_hour * sim_duration

    server_result = {
        'downtime': server_total_downtime,
        'failures': server_failures,
        'repair_cost_total': server_total_repair_cost,
        'lost_revenue': server_total_lost_revenue,
        'maintenance_cost_total': maintenance_cost_total,
        'mttr': server_mttr,
        'mttf': server_mttf,
        'availability_percent': server_availability_percent
    }

    return server_result, disks_results


def main():
    random.seed(105)
    np.random.seed(105)

    with open('config.json', 'r') as f:
        config = json.load(f)

    sim_duration = float(config['time_period_hours'])
    shape = float(config['weibull_shape'])
    currency = config['currency']
    servers = config['servers']

    all_server_results = []
    all_disk_results = []
    disk_labels = []
    server_labels = []

    total_lost_revenue = 0.0
    total_maintenance_cost = 0.0

    for server in servers:
        server_label = f"Server{server['id']}"
        server_result, disks_results = simulate_server_and_disks(sim_duration, shape, server)
        all_server_results.append((server_label, server_result))
        server_labels.append(server_label)

        total_lost_revenue += server_result['lost_revenue']
        total_maintenance_cost += server_result['maintenance_cost_total']

        for i, disk in enumerate(disks_results):
            disk_label = f"S{server['id']}D{i + 1}"
            all_disk_results.append((disk_label, disk))
            disk_labels.append(disk_label)

            total_lost_revenue += disk['lost_revenue_total']

    print("Wyniki dla dysków:")
    for label, disk in all_disk_results:
        print(f"{label}: Failures={disk['failures']}, Przestój={disk['downtime']:.2f}h, "
              f"Koszt naprawy={disk['repair_cost_total']:.2f}{currency}, "
              f"Utracone przychody={disk['lost_revenue_total']:.2f}{currency}, "
              f"MTTR={disk['mttr']:.2f}h, MTTF={disk['mttf']:.2f}h, "
              f"Dostępność={disk['availability_percent']:.2f}%")

    print("\nWyniki dla serwerów:")
    for label, server in all_server_results:
        print(
            f"{label}: Failures={server['failures']}, Przestój={server['downtime']:.2f}h, "
            f"Koszt naprawy={server['repair_cost_total']:.2f}{currency}, "
            f"Utracone przychody={server['lost_revenue']:.2f}{currency}, "
            f"Maintenance Cost={server['maintenance_cost_total']:.2f}{currency}, "
            f"MTTR={server['mttr']:.2f}h, MTTF={server['mttf']:.2f}h, "
            f"Dostępność={server['availability_percent']:.2f}%")

    csv_filename_disks = 'results_disks.csv'
    with open(csv_filename_disks, 'w', newline='') as csvfile:
        fieldnames = ['server_disk', 'downtime_hours', 'repair_cost', 'lost_revenue', 'mttr', 'mttf',
                      'availability_percent', 'failures']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for label, disk in all_disk_results:
            writer.writerow({
                'server_disk': label,
                'downtime_hours': f"{disk['downtime']:.2f}",
                'repair_cost': f"{disk['repair_cost_total']:.2f}",
                'lost_revenue': f"{disk['lost_revenue_total']:.2f}",
                'mttr': f"{disk['mttr']:.2f}",
                'mttf': f"{disk['mttf']:.2f}",
                'availability_percent': f"{disk['availability_percent']:.2f}",
                'failures': disk['failures'],
            })

    csv_filename_servers = 'results_servers.csv'
    with open(csv_filename_servers, 'w', newline='') as csvfile:
        fieldnames = ['server', 'downtime_hours', 'repair_cost', 'lost_revenue', 'maintenance_cost', 'mttr', 'mttf',
                      'availability_percent', 'failures']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for label, server in all_server_results:
            writer.writerow({
                'server': label,
                'downtime_hours': f"{server['downtime']:.2f}",
                'repair_cost': f"{server['repair_cost_total']:.2f}",
                'lost_revenue': f"{server['lost_revenue']:.2f}",
                'maintenance_cost': f"{server['maintenance_cost_total']:.2f}",
                'mttr': f"{server['mttr']:.2f}",
                'mttf': f"{server['mttf']:.2f}",
                'availability_percent': f"{server['availability_percent']:.2f}",
                'failures': server['failures'],
            })

    print(f"\nWyniki zapisane do plików {csv_filename_disks} i {csv_filename_servers}")

    downtime_values_disks = [d[1]['downtime'] for d in all_disk_results]
    repair_cost_values_disks = [d[1]['repair_cost_total'] for d in all_disk_results]
    labels_disks = [d[0] for d in all_disk_results]

    downtime_values_servers = [s[1]['downtime'] for s in all_server_results]
    repair_cost_values_servers = [s[1]['repair_cost_total'] for s in all_server_results]
    labels_servers = [s[0] for s in all_server_results]

    num_disks = len(labels_disks)
    num_servers = len(labels_servers)
    x_disks = np.arange(num_disks)
    x_servers = np.arange(num_servers)

    fig, axes = plt.subplots(3, 2, figsize=(25, 20))

    axes[0, 0].bar(x_disks, downtime_values_disks, color='tab:red')
    axes[0, 0].set_title('Czas przestoju dysków')
    axes[0, 0].set_ylabel('Godziny')
    axes[0, 0].set_xticks(x_disks)
    axes[0, 0].set_xticklabels(labels_disks, rotation=45, ha='right')

    axes[1, 0].bar(x_disks, repair_cost_values_disks, color='tab:blue')
    axes[1, 0].set_title('Koszty naprawy dysków')
    axes[1, 0].set_ylabel(f'Koszt ({currency})')
    axes[1, 0].set_xticks(x_disks)
    axes[1, 0].set_xticklabels(labels_disks, rotation=45, ha='right')

    axes[0, 1].bar(x_servers, downtime_values_servers, color='tab:purple')
    axes[0, 1].set_title('Czas przestoju serwerów')
    axes[0, 1].set_ylabel('Godziny')
    axes[0, 1].set_xticks(x_servers)
    axes[0, 1].set_xticklabels(labels_servers, rotation=45, ha='right')

    axes[1, 1].bar(x_servers, repair_cost_values_servers, color='tab:brown')
    axes[1, 1].set_title('Koszty naprawy serwerów')
    axes[1, 1].set_ylabel(f'Koszt ({currency})')
    axes[1, 1].set_xticks(x_servers)
    axes[1, 1].set_xticklabels(labels_servers, rotation=45, ha='right')

    lost_revenue_servers = [s[1]['lost_revenue'] for s in all_server_results]

    lost_revenue_disks_per_server = {f"Server{server['id']}": 0.0 for server in servers}
    for i, (label, disk) in enumerate(all_disk_results):
        server_id = label.split('D')[0][1:]
        server_key = f"Server{server_id}"
        lost_revenue_disks_per_server[server_key] += disk['lost_revenue_total']

    sorted_server_labels = sorted(lost_revenue_disks_per_server.keys(), key=lambda x: int(x[6:]))

    lost_revenue_disks_sorted = [lost_revenue_disks_per_server[label] for label in sorted_server_labels]
    lost_revenue_servers_sorted = [lost_revenue_servers[i] for i, label in enumerate(sorted_server_labels)]

    x_combined = np.arange(num_servers)
    width = 0.35

    axes[2, 0].bar(x_combined - width/2, lost_revenue_servers_sorted, width, label='Server Lost Revenue', color='tab:cyan')
    axes[2, 0].bar(x_combined + width/2, lost_revenue_disks_sorted, width, label='Disks Lost Revenue', color='tab:orange')
    axes[2, 0].set_title('Utracone dochody (Serwery i Dyski)')
    axes[2, 0].set_ylabel(f'Kwota ({currency})')
    axes[2, 0].set_xticks(x_combined)
    axes[2, 0].set_xticklabels(sorted_server_labels, rotation=45, ha='right')
    axes[2, 0].legend()

    maintenance_costs = [s[1]['maintenance_cost_total'] for s in all_server_results]
    labels_maintenance = [s[0] for s in all_server_results]
    x_maintenance = np.arange(num_servers)

    axes[2, 1].bar(x_maintenance, maintenance_costs, color='tab:green')
    axes[2, 1].set_title('Maintenance Cost for Servers')
    axes[2, 1].set_ylabel(f'Koszt ({currency})')
    axes[2, 1].set_xticks(x_maintenance)
    axes[2, 1].set_xticklabels(labels_maintenance, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('servers_disks_results.png')
    plt.show()


if __name__ == "__main__":
    main()
