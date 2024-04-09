import sh
import sys
from device_toolkit import set_logging


class RemoteRunner:

    def __init__(self,
                 cmd="sudo -E python ~/faster_rcnn/inference.py",
                 logging_path='logs/remote_runner',
                 bg=True,
                 std_out=False) -> None:
        
        self.logger = set_logging(logging_path)
        with open('logs/remote_runner.log', 'w') as f:
            f.write('') 
        self.aggregated = ""
        self.flag = True 
        self.std_out = std_out       
        print(f"Create Remote Process: {cmd}")
        self.process = sh.ssh(["nvidia@192.168.55.1", "-t", cmd], _out=self.ssh_interact, _out_bufsize=0,  _tty_in=True, _unify_ttys=True, _bg=bg, )

    def ssh_interact(self, char, stdin):
        if self.flag:
            self.aggregated += char
        if self.aggregated.endswith("password: "):
            stdin.put("nvidia\n")
            self.flag = False
        if self.std_out:
            sys.stdout.write(char)
            sys.stdout.flush()
        else:
            with open(f'{self.logger.name}.log', 'a') as f:
                f.write(char) 


if __name__ == '__main__':
    RemoteRunner(cmd='sudo -E python ~/faster_rcnn/test_fan.py 50')