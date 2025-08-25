using System.Diagnostics;
using System.Text.RegularExpressions;

namespace GomokuServer
{
    record EngineConfig(string Name, string Path, string Args, string WorkDir = "");

    class Engine(EngineConfig config)
    {
        public string Name { get; } = config.Name;
        public bool IsRunning => _process is not null && !_process.HasExited;

        readonly string _path = config.Path;
        readonly string _args = config.Args;
        readonly string _workDir = config.WorkDir;
        Process? _process;
        readonly LinkedList<(string Regex, ResponceValueFuture Value)> _waitingResponceList = new();

        public void Run()
        {
            var psi = new ProcessStartInfo
            {
                FileName = _path,
                Arguments = _args,
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true
            };

            if (!string.IsNullOrEmpty(_workDir))
                psi.WorkingDirectory = _workDir;

            _process = Process.Start(psi);
            _process!.OutputDataReceived += Process_OutputDataRecieved;
            _process.BeginOutputReadLine();
        }

        public void Kill() => _process?.Kill();

        public void WaitForExit(int timeoutMs) => _process?.WaitForExit(timeoutMs);

        public bool Quit(int timeoutMs)
        {
            if (_process is null)
                return true;

            SendCommand("quit");
            WaitForExit(timeoutMs);
            return !this._process.HasExited;
        }

        public void SetPosition(Position pos)
        {
            Span<char> colors = ['X', 'O'];
            var res = SendCommand($"pos {pos} {colors[(int)pos.SideToMove]}");
        }

        public int Think(int timeLimitMs)
        {
            var res = SendCommand($"go {timeLimitMs}", "^\\s*move\\s+\\d+");

            while (!res.HasResult && !_process!.HasExited)
                Thread.Yield();

            return int.TryParse(res.Result.Split()[1], out int move) ? move : -1;
        }

        public void SendMove(int move) => SendCommand($"move {move}");

        public void SendGameResult(DiscColor winner)
        {
            if (winner == DiscColor.Null)
            {
                SendCommand($"winner none");
                return;
            }
            
            SendCommand($"winner {winner.ToString().ToLower()}");
        }

        Responce SendCommand(string cmd, string? responceRegex = null)
        {
            Debug.WriteLine($"Server -> {_process!.ProcessName}(PID: {_process.Id}): {cmd}");

            if (responceRegex is null)
            {
                _process.StandardInput.WriteLine(cmd);
                return new Responce(cmd);
            }

            var responceFuture = new ResponceValueFuture();
            var responce = new Responce(cmd, responceFuture);
            lock (_waitingResponceList)
                _waitingResponceList.AddFirst(new LinkedListNode<(string, ResponceValueFuture)>((responceRegex, responceFuture)));
            _process.StandardInput.WriteLine(cmd);
            return responce;
        } 

        void Process_OutputDataRecieved(object sender, DataReceivedEventArgs e)
        {
            if (e.Data is null)
                return;

            Debug.WriteLine($"{_process!.ProcessName}(PID: {_process.Id}) -> Server: {e.Data}");

            lock (_waitingResponceList)
            {
                var responce = _waitingResponceList.Where(x => Regex.IsMatch(e.Data, x.Regex)).LastOrDefault();
                if (responce != default)
                {
                    _waitingResponceList.Remove(responce);
                    responce.Value.Value = e.Data;
                }
                else
                {
                    OnNonResponceCommandRecieved(e.Data);
                }
            }
        }

        void OnNonResponceCommandRecieved(string cmd)
        {
            
        }

        class Responce
        {
            public string Command { get; private set; }

            public string Result
            {
                get
                {
                    while (_result.Value is null)
                        Thread.Yield();
                    return _result.Value;
                }
            }

            public bool HasResult => _result.Value is not null;

            readonly ResponceValueFuture _result;

            public Responce(string cmd, ResponceValueFuture resultFuture)
            {
                Command = cmd;
                _result = resultFuture;
            }

            public Responce(string cmd) => (Command, _result) = (cmd, new ResponceValueFuture { Value = string.Empty });

            public bool Wait(int timeoutMs)
            {
                var start = Environment.TickCount;
                while (_result.Value is null && Environment.TickCount - start < timeoutMs)
                    Thread.Yield();
                return HasResult;
            }
        }
        
        class ResponceValueFuture { public string? Value { get; set; } = null; }
    }
}