#if ARDUINO < 100
#include <WProgram.h>                          // Add other libraries on demand as requested
#else
#include <Arduino.h>                               // Needed when working with libraries
#endif
#include <stdio.h>
/***  Declare constants and variables for the adjustable voltage regulator  ***/
const byte actualValueVoltPin          =  A1;       // Measure regulated voltage
const byte controlPin                  =  11;       // PWM for output voltage control
/***  Declare constants and variables for analog input measurement  ***/
const unsigned long refVolt            =  5000;     // Reference voltage default 5V (5000mV) to analog converter; change to 3300 if 3,3V


/////////////ADJUST VOLTAGE SETPOINT HERE///////////////////
const unsigned int outputVoltSetPoint  =  7000;     // Adjust output voltage (Set 0000 to 5000 in mV, depending on voltage divider)

///////////////////////////////////////////////////////////////////////////////////////////////////

unsigned int actualValueVolt           = 0;         // Initialize measured output voltage with 0mV to start with/***  Declare constants and variables for resistor voltage diviver at the analog input   ***/
byte PWMValue =  0;   

void setup() {                              
  pinMode(controlPin, OUTPUT);                      // Pin to control the output voltage
  digitalWrite(controlPin,HIGH);        
}

void loop() {                                    // Function Loop 
  actualValueVolt = ((refVolt * 1000) / 1023) * (analogRead(actualValueVoltPin)) / 1000;
  /***  Output voltage regulation  ***/
  if((actualValueVolt < outputVoltSetPoint)&&(PWMValue<255))  // Switch on MOSFET while increasing HIGH time of Digital Pin 6
    analogWrite(controlPin,PWMValue++);
  if((actualValueVolt > outputVoltSetPoint)&&(PWMValue>0))
    analogWrite(controlPin,PWMValue--);
}