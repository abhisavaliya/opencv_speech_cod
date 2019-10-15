import threading, time, sys

class counter :

    def num(self):
        while True:
            print("noon")
            time.sleep(.01)

    def handler(self):
        while True:
            print(' Processing .. ')
            time.sleep(0.01)





th1 = threading.Thread(target = counter().num).start()
th2 = threading.Thread(target = counter().handler).start()