import logging
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialBridge:
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        self.ser = None
        self.connected = False
        
        if not HAS_PYSERIAL:
            logging.warning("pyserial not installed. Serial communication disabled.")
            return
            
        if port is None:
            port = self._auto_detect_port()
            
        if port:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino reset
                self.connected = True
                logging.info(f"Connected to Arduino on {port} at {baudrate} baud.")
            except Exception as e:
                logging.warning(f"Failed to connect to Arduino on {port}: {e}")
        else:
            logging.warning("No Arduino port provided or found.")

    def _auto_detect_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "Arduino" in port.description or "CH340" in port.description:
                return port.device
        return None

    def is_connected(self) -> bool:
        return self.connected

    def send(self, command: str) -> bool:
        if not self.connected or self.ser is None:
            logging.debug(f"Simulating serial send (not connected): {command}")
            return False
            
        try:
            cmd_str = f"{command.strip()}\n"
            self.ser.write(cmd_str.encode('utf-8'))
            return True
        except Exception as e:
            logging.error(f"Error sending to serial: {e}")
            self.connected = False
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            logging.info("Serial port closed.")

if __name__ == "__main__":
    bridge = SerialBridge()
    if bridge.is_connected():
        bridge.send("PASS")
        time.sleep(2)
        bridge.send("REJECT")
        time.sleep(1)
        bridge.send("RESET")
        bridge.close()
