# Fingertip Keyboard Input
A computer application that allows keyboard input by drawing letters in the air, captured by a web camera.

## Gesture Mapping
Subject to change
| Command    | Gesture 1 | Gesture 2    |
| ---------- | --------- | ------------ |
| Draw       | Right 1   |              |
| Submit     | Right Fist|              |
| Clear      | Right 4   |              |
| Backspace  | Left 1    | On-screen Box|

Note that the on-screen box activates based on the position of the center of the palm.

## How To Run
Within the `fingertip_keyboard_input` directory, run `python fingertip_keyboard_input.py`

## Todo
[ ] Option to choose character if model is unsure of which letter (ex. 'C' vs. 'c', 'K' vs. 'k')
[ ] Integrate with existing two_handed_gestures mouse implementation
[ ] Path smoothing
[ ] Gesture smoothing