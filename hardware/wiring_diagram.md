# Morphology-Aware Crack Inspection - Hardware Wiring Guide

This guide details the pin connections for the Arduino Uno handling the physical REJECT/PASS mechanisms.

## Components Needed
1. Arduino Uno
2. SG90 Micro Servo
3. 5mm Red LED
4. 5mm Green LED
5. 5V Active Buzzer
6. 2x 220Ω Resistors
7. Breadboard & Jumper Wires

## ASCII Circuit Schematic

```text
USB Power (from PC) ----> Arduino Uno [5V] ----> Breadboard [+] Rail
                          Arduino Uno [GND] ---> Breadboard [-] Rail

[ Red LED ]
Pin 2  ---> [220Ω Resistor] ---> Anode (+)
                                 Cathode (-) ---> [-] Rail

[ Green LED ]
Pin 3  ---> [220Ω Resistor] ---> Anode (+)
                                 Cathode (-) ---> [-] Rail

[ Buzzer ]
Pin 4  ---> Positive (+) terminal
            Negative (-) terminal ---> [-] Rail

[ SG90 Servo ]
Pin 9  ---> Control Wire (Orange/Yellow)
[5V]   ---> VCC Wire (Red)
[GND]  ---> GND Wire (Brown/Black)
```

## Pin-by-Pin Guide for Pulkit

*   **Digital Pin 2**: Connects to the Red LED (via resistor). Indicates a `REJECT` state.
*   **Digital Pin 3**: Connects to the Green LED (via resistor). Indicates a `PASS` state.
*   **Digital Pin 4**: Connects to the positive leg of the active buzzer. Sounds when `REJECT` is active.
*   **Digital Pin 9 (PWM)**: Connects to the signal line of the SG90 Servo. Sweeps to push defective items.
*   **5V & GND**: Ensure all components share the same ground. The SG90 can be powered directly from the Arduino 5V pin as long as it's not under heavy load. If you add more servos later, use an external 5V supply.

*Note: The Arduino receives power and serial commands via USB from the inference PC.*
