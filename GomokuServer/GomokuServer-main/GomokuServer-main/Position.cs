using System.Collections.ObjectModel;

namespace GomokuServer
{
    enum DiscColor
    {
        Black = 0,
        White = 1,
        Null = 2
    }

    class Position
    {
        const int StoneCountForWin = 5;
        const int BoardSizeMin = StoneCountForWin;
        const int BoardSizeMax = 19;
        const int OutOfBoard = 3;
        const int Margin = 1;
        readonly static ReadOnlyCollection<int> SqrtTable;
        static ReadOnlySpan<char> StoneChars => ['X', 'O', '-'];

        public int BoardSize { get; }
        public int PaddedBoardSize { get; }
        public DiscColor SideToMove { get; private set; }
        public DiscColor OpponentColor => SideToMove ^ DiscColor.White;
        public DiscColor Winner { get; private set; } = DiscColor.Null;
        public DiscColor Loser => (Winner == DiscColor.Null) ? DiscColor.Null : Winner ^ DiscColor.White;
        public bool IsFull => _board.All(x => x != (int)DiscColor.Null);

        readonly int[] _board;

        static Position()
        {
            var sqrt = new int[BoardSizeMax * BoardSizeMax + 1];
            for (var i = 0; i <= BoardSizeMax; i++)
                sqrt[i * i] = i;

            for (var i = 1; i < sqrt.Length; i++)
            {
                if (sqrt[i] == 0)
                    sqrt[i] = -1;
            }

            SqrtTable = new ReadOnlyCollection<int>(sqrt);
        }

        public static Position? TryParse(string str)
        {
            var splitted = str.Split();
            var (board, side) = (splitted[0], splitted[1]);

            if (board.Length > BoardSizeMax * BoardSizeMax)
                return null;

            var size = SqrtTable[board.Length];

            if (size == -1)
                return null;

            var pos = new Position(size);
            for (var i = 0; i < board.Length; i++)
            {
                if (board[i] == StoneChars[0])
                    pos.SetStoneAt(i, DiscColor.Black);
                else if (board[i] == StoneChars[1])
                    pos.SetStoneAt(i, DiscColor.White);
                else if (board[i] != StoneChars[2])
                    return null;
            }

            if (side[0] == StoneChars[0])
                pos.SideToMove = DiscColor.Black;
            else if (side[0] == StoneChars[1])
                pos.SideToMove = DiscColor.White;
            else
                return null;

            return pos;
        }

        public Position(int size)
        {
            BoardSize = size;
            PaddedBoardSize = size + Margin * 2;
            _board = new int[PaddedBoardSize * PaddedBoardSize];
            Array.Fill(_board, (int)DiscColor.Null);

            for (var i = 0; i < PaddedBoardSize; i++)
            {
                _board[i] = OutOfBoard;
                _board[^(PaddedBoardSize - i)] = OutOfBoard;
            }

            for (var i = 0; i < BoardSize; i++)
            {
                var y = Margin + i;
                for (var j = 0; j < Margin; j++)
                    _board[y * PaddedBoardSize + j] = _board[(y + 1) * PaddedBoardSize - j - 1] = OutOfBoard;
            }
        }

        public Position(Position pos) : this(pos.BoardSize) => pos.CopyTo(this);

        public void DebugPrint()
        {
#if DEBUG
            for (var y = 0; y < BoardSize; y++)
            {
                for (var x = 0; x < BoardSize; x++)
                    Console.Write(StoneChars[(int)GetStoneAt(x + y * BoardSize)]);
                Console.WriteLine();
            }
            Console.WriteLine();
#endif
        }

        public void CopyTo(Position dest)
        {
            if (dest.BoardSize != BoardSize)
                throw new ArgumentException($"The board size does not match: expected {BoardSize} but got {dest.BoardSize}");

            dest.SideToMove = SideToMove;
            dest.Winner = Winner;
            Array.Copy(_board, dest._board, _board.Length);
        }

        public DiscColor GetStoneAt(int coord)
        {
            var (x, y) = (coord % BoardSize, coord / BoardSize);
            x += Margin;
            y += Margin;
            var s = _board[x + y * PaddedBoardSize];
            return (s == OutOfBoard) ? DiscColor.Null : (DiscColor)s;
        }

        public void SetStoneAt(int coord, DiscColor color)
        {
            var (x, y) = (coord % BoardSize, coord / BoardSize);
            x += Margin;
            y += Margin;
            _board[x + y * PaddedBoardSize] = (int)color;
        }

        public bool Update(int coord)
        {
            if (coord < 0 || coord >= BoardSize * BoardSize)
                return false;
                
            var (x, y) = (coord % BoardSize, coord / BoardSize);
            x += Margin;
            y += Margin;
            coord = x + y * PaddedBoardSize;

            if (_board[coord] != (int)DiscColor.Null)
                return false;

            Span<int> dirs = [1, PaddedBoardSize, PaddedBoardSize + 1, PaddedBoardSize - 1];
            foreach (var dir in dirs)
            {
                var count = 0;
                int i;

                i = coord;
                while (_board[i += dir] == (int)SideToMove)
                    count++;

                i = coord;
                while (_board[i -= dir] == (int)SideToMove)
                    count++;

                if (count + 1 >= StoneCountForWin)
                {
                    Winner = SideToMove;
                    break;
                }
            }

            _board[coord] = (int)SideToMove;
            this.SideToMove = this.OpponentColor;

            return true;
        }

        public string CoordinateToString(int move)
        {
            var x = (char)('A' + move % BoardSize);
            var y = BoardSize - move / BoardSize;
            return $"{x}{y}";
        }

        public override string ToString()
        {
            Span<char> boardStr = stackalloc char[BoardSize * BoardSize];
            var i = 0;
            for (var j = 0; j < _board.Length; j++)
            {
                if (_board[j] == OutOfBoard)
                    continue;
                boardStr[i++] = StoneChars[_board[j]];
            }
            return new string(boardStr);
        }
    }
}