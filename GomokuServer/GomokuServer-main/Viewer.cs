using System.Diagnostics;
using System.Text.RegularExpressions;

namespace GomokuServer
{
    class Viewer(string path, string args)
    {
        public bool IsRunning => _process is not null && !_process.HasExited;

        readonly string _path = path;
        readonly string _args = args;
        Process? _process;
        volatile bool _waitForOK = false;

        public void Run()
        {
            var psi = new ProcessStartInfo
            {
                FileName = _path,
                Arguments = _args,
                CreateNoWindow = false,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true
            };

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
            SendCommand($"pos {pos} {colors[(int)pos.SideToMove]}");
        }

        public void SendGameResult(DiscColor winner)
        {
            _waitForOK = true;
            if (winner == DiscColor.Null)
                SendCommand($"winner none");
            else
                SendCommand($"winner {winner.ToString().ToLower()}");

            while (_waitForOK)
                Thread.Yield();
        }

        void SendCommand(string cmd, string? responceRegex = null)
        {
            Debug.WriteLine($"Server -> {_process!.ProcessName}(PID: {_process.Id}): {cmd}");
            _process.StandardInput.WriteLine(cmd);
        }

        void Process_OutputDataRecieved(object sender, DataReceivedEventArgs e)
        {
            if (e.Data is null)
                return;

            Debug.WriteLine($"{_process!.ProcessName}(PID: {_process.Id}) -> Server: {e.Data}");

            if (e.Data == "ok")
                _waitForOK = false;
        }
    }
}