Use skill `libraries-analysis-skill`.

# Task

Develop Two Applications Using the Cengal Library.

Develop the following two applications:

## 1. Asynchronous wxPython Frontend
Develop an asynchronous `wxPython` application with the following functionality:
* **Code Execution:** Provide a text input area for Python functions. Upon clicking a button, send the input code to the backend for execution.
* **Result Display:** Display the execution results from the backend to the user.
* **Connection Management:** 
    * Display the current connection status (Connected/Disconnected).
    * If the connection is lost or initially unavailable, provide a "Connect" button to attempt reconnection to the backend.
* **Technical Requirements:**
    * Use `Cengal`'s asynchronous `wxPython` implementation.
    * Use `Cengal`'s custom `asyncio.streams` replacements for all asynchronous backend communication.

## 2. Asynchronous TUI Backend
Develop an asynchronous TUI (Terminal User Interface) backend application with the following architecture and features:
* **TUI Interface:** Use the `terminal` module from the `Cengal` library. The interface must include:
    * **Controls:** A selectable menu with two options navigable via arrow keys:
        1. Disconnect from the frontend.
        2. Exit application.
    * **Status Dashboard:** Real-time display of:
        * Connection status (Frontend connected/disconnected).
        * Current task status (Idle/Running).
        * Execution time (if a task is currently running).
        * Task history (including: start time, total duration, input code size, and output result size).
* **Architecture:**
    * **Main Process:** Manages the TUI and handles all communication with the frontend.
    * **Worker Process:** A secondary process that executes a queue of tasks received from the main process.
    * **Inter-Process Communication (IPC):** You must use `Cengal`'s `fast shared memory based inter-process-communication module for asyncio` for communication between the main process and the worker process.
