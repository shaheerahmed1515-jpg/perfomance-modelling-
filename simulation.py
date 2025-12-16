

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.csv"

SCENARIOS = [1, 2, 3]  # number of doctors to compare


def simulate(df: pd.DataFrame, num_doctors: int):
    """Simulate FCFS queue with num_doctors servers."""
    avail = [0.0] * num_doctors     # next available time per doctor
    busy = [0.0] * num_doctors      # total busy time per doctor

    rows = []
    for _, r in df.iterrows():
        arrival = float(r["ArrivalTime_min"])
        service = float(r["ServiceTime_min"])

        # assign the patient to the earliest available doctor
        d_idx = int(np.argmin(avail))
        start = max(arrival, avail[d_idx])
        wait = start - arrival
        finish = start + service

        avail[d_idx] = finish
        busy[d_idx] += service

        rows.append({
            "PatientID": r["PatientID"],
            "ArrivalTime_min": arrival,
            "ServiceTime_min": service,
            "Doctor": f"D{d_idx + 1}",
            "StartService_min": start,
            "WaitingTime_min": wait,
            "FinishTime_min": finish
        })

    detail = pd.DataFrame(rows)
    sim_end = float(detail["FinishTime_min"].max())

    avg_wait = float(detail["WaitingTime_min"].mean())
    throughput = len(detail) / (sim_end / 60.0) if sim_end > 0 else float("nan")

    total_busy = float(sum(busy))
    utilization = total_busy / (num_doctors * sim_end) if sim_end > 0 else float("nan")

    # Simple estimated average queue length:
    # (Total waiting time area) / (simulation time)
    avg_queue_len = float(detail["WaitingTime_min"].sum() / sim_end) if sim_end > 0 else float("nan")

    metrics = {
        "Doctors": num_doctors,
        "AvgWaitingTime_min": avg_wait,
        "AvgQueueLength_est": avg_queue_len,
        "Throughput_patients_per_hr": throughput,
        "DoctorUtilization_fraction": utilization,
        "DoctorUtilization_percent": utilization * 100,
        "SimulationEnd_min": sim_end,
    }
    return detail, metrics


def plot_line(x, y, xlabel, ylabel, title, out_path: Path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATASET_PATH)

    summary_rows = []
    for d in SCENARIOS:
        detail, metrics = simulate(df, d)
        detail.to_csv(BASE_DIR / f"results_detail_{d}doctors.csv", index=False)
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE_DIR / "results_summary.csv", index=False)

    # Create graphs
    plot_line(summary["Doctors"], summary["AvgWaitingTime_min"],
              "Number of Doctors", "Average Waiting Time (minutes)",
              "Average Waiting Time vs Number of Doctors", BASE_DIR / "avg_wait.png")

    plot_line(summary["Doctors"], summary["AvgQueueLength_est"],
              "Number of Doctors", "Average Queue Length (estimated)",
              "Average Queue Length vs Number of Doctors", BASE_DIR / "avg_queue.png")

    plot_line(summary["Doctors"], summary["DoctorUtilization_percent"],
              "Number of Doctors", "Doctor Utilization (%)",
              "Doctor Utilization vs Number of Doctors", BASE_DIR / "utilization.png")

    plot_line(summary["Doctors"], summary["Throughput_patients_per_hr"],
              "Number of Doctors", "Throughput (patients/hour)",
              "Throughput vs Number of Doctors", BASE_DIR / "throughput.png")

    print("Done!")
    print("Created:")
    print("- dataset.csv")
    print("- results_summary.csv")
    print("- results_detail_{N}doctors.csv")
    print("- avg_wait.png, avg_queue.png, utilization.png, throughput.png")


if __name__ == "__main__":
    main()
