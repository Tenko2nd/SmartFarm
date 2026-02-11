import csv
import os
import socket
import time

# SERVER CONSTANT
SERVER_ADDRESS = ('udpserver.bu.ac.th', 5005)
BUFFER_SIZE = 1024

# DATABASE CONSTANT
CSV_FILENAME = "sensor_data.csv"
COLUMNS = ['Timestamp', 'Temperature', 'Humidity', 'Moisture', 'Light', 'Gas']
INTERVAL = 10

def send_command(command, id, data=None):
    response = None
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    try:
        if command.upper() == 'SET' and data is not None:
            message = f"SET,{id},{data}"
        elif command.upper() == 'GET':
            message = f"GET,{id}"
        else:
            print("Invalid command or missing data for SET.")
            return

        UDPClientSocket.sendto(str.encode(message), SERVER_ADDRESS)
        response, _ = UDPClientSocket.recvfrom(BUFFER_SIZE)
        print("Server response:", response.decode())
    finally:
        UDPClientSocket.close()

    if response is not None:
        return response.decode()
    return None


def parse_data(raw_string):
    data_dict = {}
    parts = raw_string.split(';')

    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()

            if value.upper() == "NO DATA":
                data_dict[key] = ""
            else:
                try:
                    data_dict[key] = float(value)
                except ValueError:
                    data_dict[key] = value

    return data_dict


def save_row(data):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()

        row = {k: data.get(k, "") for k in COLUMNS}
        writer.writerow(row)


def main(interval_seconds=10):
    last_timestamp = None
    print(f"Démarrage de la collecte (toutes les {interval_seconds}s)... Ctrl+C pour arrêter.")
    try:
        while True:
            raw_response = send_command('GET', 'car')
            data = parse_data(raw_response)

            current_timestamp = data.get('Timestamp')

            if current_timestamp is None:
                pass
            elif current_timestamp == last_timestamp:
                pass
            else:
                save_row(data)
                last_timestamp = current_timestamp
                print(f"Enregistré : {current_timestamp}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nArrêt de la collecte.")


if __name__ == "__main__":
    main(interval_seconds=10)