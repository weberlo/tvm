# Generated from /Users/doobs/Code/repo/sampl/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys



def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\63")
        buf.write("\u0164\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%")
        buf.write("\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4,\t,\4-\t-\4.")
        buf.write("\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64")
        buf.write("\t\64\4\65\t\65\4\66\t\66\4\67\t\67\3\2\3\2\3\3\3\3\3")
        buf.write("\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\n\3\n")
        buf.write("\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\f\3\r\3\r\3\16\3\16")
        buf.write("\3\16\3\16\3\17\3\17\3\20\3\20\3\21\3\21\3\21\3\22\3\22")
        buf.write("\3\22\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3\25\3\25\3\25")
        buf.write("\3\25\3\25\3\26\3\26\3\27\3\27\3\27\3\30\3\30\3\30\3\30")
        buf.write("\3\30\3\30\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\32\3\32")
        buf.write("\3\33\3\33\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\35\3\35")
        buf.write("\3\35\3\35\3\35\3\36\3\36\3\36\3\36\3\36\3\36\3\36\3\37")
        buf.write("\3\37\3\37\3\37\3\37\7\37\u00d4\n\37\f\37\16\37\u00d7")
        buf.write("\13\37\3\37\3\37\3\37\3\37\3\37\3 \6 \u00df\n \r \16 ")
        buf.write("\u00e0\3 \3 \3!\3!\3!\3!\7!\u00e9\n!\f!\16!\u00ec\13!")
        buf.write("\3!\3!\3!\3!\3\"\3\"\3\"\3#\3#\3#\7#\u00f8\n#\f#\16#\u00fb")
        buf.write("\13#\3#\3#\3$\3$\3%\3%\3&\3&\3\'\3\'\3(\3(\3)\3)\3*\3")
        buf.write("*\3*\3+\3+\3+\3,\3,\3,\3-\3-\3-\3.\3.\3.\3.\3.\3.\3.\3")
        buf.write(".\3.\5.\u0120\n.\3/\3/\5/\u0124\n/\3/\3/\3/\7/\u0129\n")
        buf.write("/\f/\16/\u012c\13/\3/\3/\7/\u0130\n/\f/\16/\u0133\13/")
        buf.write("\3\60\3\60\3\60\3\60\3\60\3\60\3\61\3\61\3\61\5\61\u013e")
        buf.write("\n\61\3\61\5\61\u0141\n\61\3\62\3\62\3\62\3\63\6\63\u0147")
        buf.write("\n\63\r\63\16\63\u0148\3\64\3\64\5\64\u014d\n\64\3\64")
        buf.write("\3\64\3\65\3\65\3\66\3\66\3\67\3\67\3\67\3\67\3\67\3\67")
        buf.write("\3\67\3\67\3\67\3\67\3\67\7\67\u0160\n\67\f\67\16\67\u0163")
        buf.write("\13\67\5\u00d5\u00ea\u00f9\28\3\3\5\4\7\5\t\6\13\7\r\b")
        buf.write("\17\t\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22")
        buf.write("#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33\65\34\67\35")
        buf.write("9\36;\37= ?!A\"C\2E#G$I%K&M\'O(Q)S*U+W,Y-[.]/_\60a\2c")
        buf.write("\61e\62g\2i\2k\2m\63\3\2\b\5\2\13\f\17\17\"\"\4\2\f\f")
        buf.write("\17\17\4\2GGgg\4\2--//\4\2C\\c|\3\2\62;\2\u016f\2\3\3")
        buf.write("\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2")
        buf.write("\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23\3\2\2\2")
        buf.write("\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2")
        buf.write("\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2")
        buf.write("\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3")
        buf.write("\2\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2\2\2\67\3\2")
        buf.write("\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2\2\2\2A\3")
        buf.write("\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2\2\2M")
        buf.write("\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2")
        buf.write("W\3\2\2\2\2Y\3\2\2\2\2[\3\2\2\2\2]\3\2\2\2\2_\3\2\2\2")
        buf.write("\2c\3\2\2\2\2e\3\2\2\2\2m\3\2\2\2\3o\3\2\2\2\5q\3\2\2")
        buf.write("\2\7s\3\2\2\2\tu\3\2\2\2\13w\3\2\2\2\ry\3\2\2\2\17{\3")
        buf.write("\2\2\2\21}\3\2\2\2\23\177\3\2\2\2\25\u0081\3\2\2\2\27")
        buf.write("\u0084\3\2\2\2\31\u0089\3\2\2\2\33\u008b\3\2\2\2\35\u008f")
        buf.write("\3\2\2\2\37\u0091\3\2\2\2!\u0093\3\2\2\2#\u0096\3\2\2")
        buf.write("\2%\u0099\3\2\2\2\'\u009c\3\2\2\2)\u00a0\3\2\2\2+\u00a5")
        buf.write("\3\2\2\2-\u00a7\3\2\2\2/\u00aa\3\2\2\2\61\u00b0\3\2\2")
        buf.write("\2\63\u00b7\3\2\2\2\65\u00b9\3\2\2\2\67\u00bb\3\2\2\2")
        buf.write("9\u00c2\3\2\2\2;\u00c7\3\2\2\2=\u00ce\3\2\2\2?\u00de\3")
        buf.write("\2\2\2A\u00e4\3\2\2\2C\u00f1\3\2\2\2E\u00f4\3\2\2\2G\u00fe")
        buf.write("\3\2\2\2I\u0100\3\2\2\2K\u0102\3\2\2\2M\u0104\3\2\2\2")
        buf.write("O\u0106\3\2\2\2Q\u0108\3\2\2\2S\u010a\3\2\2\2U\u010d\3")
        buf.write("\2\2\2W\u0110\3\2\2\2Y\u0113\3\2\2\2[\u011f\3\2\2\2]\u0123")
        buf.write("\3\2\2\2_\u0134\3\2\2\2a\u013a\3\2\2\2c\u0142\3\2\2\2")
        buf.write("e\u0146\3\2\2\2g\u014a\3\2\2\2i\u0150\3\2\2\2k\u0152\3")
        buf.write("\2\2\2m\u0154\3\2\2\2op\7B\2\2p\4\3\2\2\2qr\7\'\2\2r\6")
        buf.write("\3\2\2\2st\7.\2\2t\b\3\2\2\2uv\7*\2\2v\n\3\2\2\2wx\7+")
        buf.write("\2\2x\f\3\2\2\2yz\7}\2\2z\16\3\2\2\2{|\7\177\2\2|\20\3")
        buf.write("\2\2\2}~\7]\2\2~\22\3\2\2\2\177\u0080\7_\2\2\u0080\24")
        buf.write("\3\2\2\2\u0081\u0082\7k\2\2\u0082\u0083\7h\2\2\u0083\26")
        buf.write("\3\2\2\2\u0084\u0085\7g\2\2\u0085\u0086\7n\2\2\u0086\u0087")
        buf.write("\7u\2\2\u0087\u0088\7g\2\2\u0088\30\3\2\2\2\u0089\u008a")
        buf.write("\7\60\2\2\u008a\32\3\2\2\2\u008b\u008c\7n\2\2\u008c\u008d")
        buf.write("\7g\2\2\u008d\u008e\7v\2\2\u008e\34\3\2\2\2\u008f\u0090")
        buf.write("\7?\2\2\u0090\36\3\2\2\2\u0091\u0092\7=\2\2\u0092 \3\2")
        buf.write("\2\2\u0093\u0094\7=\2\2\u0094\u0095\7=\2\2\u0095\"\3\2")
        buf.write("\2\2\u0096\u0097\7h\2\2\u0097\u0098\7p\2\2\u0098$\3\2")
        buf.write("\2\2\u0099\u009a\7/\2\2\u009a\u009b\7@\2\2\u009b&\3\2")
        buf.write("\2\2\u009c\u009d\7f\2\2\u009d\u009e\7g\2\2\u009e\u009f")
        buf.write("\7h\2\2\u009f(\3\2\2\2\u00a0\u00a1\7v\2\2\u00a1\u00a2")
        buf.write("\7{\2\2\u00a2\u00a3\7r\2\2\u00a3\u00a4\7g\2\2\u00a4*\3")
        buf.write("\2\2\2\u00a5\u00a6\7~\2\2\u00a6,\3\2\2\2\u00a7\u00a8\7")
        buf.write("?\2\2\u00a8\u00a9\7@\2\2\u00a9.\3\2\2\2\u00aa\u00ab\7")
        buf.write("o\2\2\u00ab\u00ac\7c\2\2\u00ac\u00ad\7v\2\2\u00ad\u00ae")
        buf.write("\7e\2\2\u00ae\u00af\7j\2\2\u00af\60\3\2\2\2\u00b0\u00b1")
        buf.write("\7o\2\2\u00b1\u00b2\7c\2\2\u00b2\u00b3\7v\2\2\u00b3\u00b4")
        buf.write("\7e\2\2\u00b4\u00b5\7j\2\2\u00b5\u00b6\7A\2\2\u00b6\62")
        buf.write("\3\2\2\2\u00b7\u00b8\7a\2\2\u00b8\64\3\2\2\2\u00b9\u00ba")
        buf.write("\7<\2\2\u00ba\66\3\2\2\2\u00bb\u00bc\7V\2\2\u00bc\u00bd")
        buf.write("\7g\2\2\u00bd\u00be\7p\2\2\u00be\u00bf\7u\2\2\u00bf\u00c0")
        buf.write("\7q\2\2\u00c0\u00c1\7t\2\2\u00c18\3\2\2\2\u00c2\u00c3")
        buf.write("\7o\2\2\u00c3\u00c4\7g\2\2\u00c4\u00c5\7v\2\2\u00c5\u00c6")
        buf.write("\7c\2\2\u00c6:\3\2\2\2\u00c7\u00c8\7x\2\2\u00c8\u00c9")
        buf.write("\7\62\2\2\u00c9\u00ca\7\60\2\2\u00ca\u00cb\7\62\2\2\u00cb")
        buf.write("\u00cc\7\60\2\2\u00cc\u00cd\7\65\2\2\u00cd<\3\2\2\2\u00ce")
        buf.write("\u00cf\7\61\2\2\u00cf\u00d0\7,\2\2\u00d0\u00d5\3\2\2\2")
        buf.write("\u00d1\u00d4\5=\37\2\u00d2\u00d4\13\2\2\2\u00d3\u00d1")
        buf.write("\3\2\2\2\u00d3\u00d2\3\2\2\2\u00d4\u00d7\3\2\2\2\u00d5")
        buf.write("\u00d6\3\2\2\2\u00d5\u00d3\3\2\2\2\u00d6\u00d8\3\2\2\2")
        buf.write("\u00d7\u00d5\3\2\2\2\u00d8\u00d9\7,\2\2\u00d9\u00da\7")
        buf.write("\61\2\2\u00da\u00db\3\2\2\2\u00db\u00dc\b\37\2\2\u00dc")
        buf.write(">\3\2\2\2\u00dd\u00df\t\2\2\2\u00de\u00dd\3\2\2\2\u00df")
        buf.write("\u00e0\3\2\2\2\u00e0\u00de\3\2\2\2\u00e0\u00e1\3\2\2\2")
        buf.write("\u00e1\u00e2\3\2\2\2\u00e2\u00e3\b \2\2\u00e3@\3\2\2\2")
        buf.write("\u00e4\u00e5\7\61\2\2\u00e5\u00e6\7\61\2\2\u00e6\u00ea")
        buf.write("\3\2\2\2\u00e7\u00e9\13\2\2\2\u00e8\u00e7\3\2\2\2\u00e9")
        buf.write("\u00ec\3\2\2\2\u00ea\u00eb\3\2\2\2\u00ea\u00e8\3\2\2\2")
        buf.write("\u00eb\u00ed\3\2\2\2\u00ec\u00ea\3\2\2\2\u00ed\u00ee\7")
        buf.write("\f\2\2\u00ee\u00ef\3\2\2\2\u00ef\u00f0\b!\2\2\u00f0B\3")
        buf.write("\2\2\2\u00f1\u00f2\7^\2\2\u00f2\u00f3\7$\2\2\u00f3D\3")
        buf.write("\2\2\2\u00f4\u00f9\7$\2\2\u00f5\u00f8\5C\"\2\u00f6\u00f8")
        buf.write("\n\3\2\2\u00f7\u00f5\3\2\2\2\u00f7\u00f6\3\2\2\2\u00f8")
        buf.write("\u00fb\3\2\2\2\u00f9\u00fa\3\2\2\2\u00f9\u00f7\3\2\2\2")
        buf.write("\u00fa\u00fc\3\2\2\2\u00fb\u00f9\3\2\2\2\u00fc\u00fd\7")
        buf.write("$\2\2\u00fdF\3\2\2\2\u00fe\u00ff\7,\2\2\u00ffH\3\2\2\2")
        buf.write("\u0100\u0101\7\61\2\2\u0101J\3\2\2\2\u0102\u0103\7-\2")
        buf.write("\2\u0103L\3\2\2\2\u0104\u0105\7/\2\2\u0105N\3\2\2\2\u0106")
        buf.write("\u0107\7>\2\2\u0107P\3\2\2\2\u0108\u0109\7@\2\2\u0109")
        buf.write("R\3\2\2\2\u010a\u010b\7>\2\2\u010b\u010c\7?\2\2\u010c")
        buf.write("T\3\2\2\2\u010d\u010e\7@\2\2\u010e\u010f\7?\2\2\u010f")
        buf.write("V\3\2\2\2\u0110\u0111\7?\2\2\u0111\u0112\7?\2\2\u0112")
        buf.write("X\3\2\2\2\u0113\u0114\7#\2\2\u0114\u0115\7?\2\2\u0115")
        buf.write("Z\3\2\2\2\u0116\u0117\7V\2\2\u0117\u0118\7t\2\2\u0118")
        buf.write("\u0119\7w\2\2\u0119\u0120\7g\2\2\u011a\u011b\7H\2\2\u011b")
        buf.write("\u011c\7c\2\2\u011c\u011d\7n\2\2\u011d\u011e\7u\2\2\u011e")
        buf.write("\u0120\7g\2\2\u011f\u0116\3\2\2\2\u011f\u011a\3\2\2\2")
        buf.write("\u0120\\\3\2\2\2\u0121\u0124\7a\2\2\u0122\u0124\5i\65")
        buf.write("\2\u0123\u0121\3\2\2\2\u0123\u0122\3\2\2\2\u0124\u012a")
        buf.write("\3\2\2\2\u0125\u0129\7a\2\2\u0126\u0129\5i\65\2\u0127")
        buf.write("\u0129\5k\66\2\u0128\u0125\3\2\2\2\u0128\u0126\3\2\2\2")
        buf.write("\u0128\u0127\3\2\2\2\u0129\u012c\3\2\2\2\u012a\u0128\3")
        buf.write("\2\2\2\u012a\u012b\3\2\2\2\u012b\u0131\3\2\2\2\u012c\u012a")
        buf.write("\3\2\2\2\u012d\u012e\7\60\2\2\u012e\u0130\5]/\2\u012f")
        buf.write("\u012d\3\2\2\2\u0130\u0133\3\2\2\2\u0131\u012f\3\2\2\2")
        buf.write("\u0131\u0132\3\2\2\2\u0132^\3\2\2\2\u0133\u0131\3\2\2")
        buf.write("\2\u0134\u0135\7k\2\2\u0135\u0136\7p\2\2\u0136\u0137\7")
        buf.write("v\2\2\u0137\u0138\78\2\2\u0138\u0139\7\66\2\2\u0139`\3")
        buf.write("\2\2\2\u013a\u013d\5e\63\2\u013b\u013c\7\60\2\2\u013c")
        buf.write("\u013e\5e\63\2\u013d\u013b\3\2\2\2\u013d\u013e\3\2\2\2")
        buf.write("\u013e\u0140\3\2\2\2\u013f\u0141\5g\64\2\u0140\u013f\3")
        buf.write("\2\2\2\u0140\u0141\3\2\2\2\u0141b\3\2\2\2\u0142\u0143")
        buf.write("\5a\61\2\u0143\u0144\7h\2\2\u0144d\3\2\2\2\u0145\u0147")
        buf.write("\5k\66\2\u0146\u0145\3\2\2\2\u0147\u0148\3\2\2\2\u0148")
        buf.write("\u0146\3\2\2\2\u0148\u0149\3\2\2\2\u0149f\3\2\2\2\u014a")
        buf.write("\u014c\t\4\2\2\u014b\u014d\t\5\2\2\u014c\u014b\3\2\2\2")
        buf.write("\u014c\u014d\3\2\2\2\u014d\u014e\3\2\2\2\u014e\u014f\5")
        buf.write("e\63\2\u014fh\3\2\2\2\u0150\u0151\t\6\2\2\u0151j\3\2\2")
        buf.write("\2\u0152\u0153\t\7\2\2\u0153l\3\2\2\2\u0154\u0155\7O\2")
        buf.write("\2\u0155\u0156\7G\2\2\u0156\u0157\7V\2\2\u0157\u0158\7")
        buf.write("C\2\2\u0158\u0159\7F\2\2\u0159\u015a\7C\2\2\u015a\u015b")
        buf.write("\7V\2\2\u015b\u015c\7C\2\2\u015c\u015d\7<\2\2\u015d\u0161")
        buf.write("\3\2\2\2\u015e\u0160\13\2\2\2\u015f\u015e\3\2\2\2\u0160")
        buf.write("\u0163\3\2\2\2\u0161\u015f\3\2\2\2\u0161\u0162\3\2\2\2")
        buf.write("\u0162n\3\2\2\2\u0163\u0161\3\2\2\2\23\2\u00d3\u00d5\u00e0")
        buf.write("\u00ea\u00f7\u00f9\u011f\u0123\u0128\u012a\u0131\u013d")
        buf.write("\u0140\u0148\u014c\u0161\3\b\2\2")
        return buf.getvalue()


class RelayLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    T__18 = 19
    T__19 = 20
    T__20 = 21
    T__21 = 22
    T__22 = 23
    T__23 = 24
    T__24 = 25
    T__25 = 26
    T__26 = 27
    T__27 = 28
    SEMVER = 29
    COMMENT = 30
    WS = 31
    LINE_COMMENT = 32
    QUOTED_STRING = 33
    MUL = 34
    DIV = 35
    ADD = 36
    SUB = 37
    LT = 38
    GT = 39
    LE = 40
    GE = 41
    EQ = 42
    NE = 43
    BOOL_LIT = 44
    CNAME = 45
    DATATYPE = 46
    FLOAT = 47
    NAT = 48
    METADATA = 49

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'@'", "'%'", "','", "'('", "')'", "'{'", "'}'", "'['", "']'", 
            "'if'", "'else'", "'.'", "'let'", "'='", "';'", "';;'", "'fn'", 
            "'->'", "'def'", "'type'", "'|'", "'=>'", "'match'", "'match?'", 
            "'_'", "':'", "'Tensor'", "'meta'", "'v0.0.3'", "'*'", "'/'", 
            "'+'", "'-'", "'<'", "'>'", "'<='", "'>='", "'=='", "'!='", 
            "'int64'" ]

    symbolicNames = [ "<INVALID>",
            "SEMVER", "COMMENT", "WS", "LINE_COMMENT", "QUOTED_STRING", 
            "MUL", "DIV", "ADD", "SUB", "LT", "GT", "LE", "GE", "EQ", "NE", 
            "BOOL_LIT", "CNAME", "DATATYPE", "FLOAT", "NAT", "METADATA" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13", 
                  "T__14", "T__15", "T__16", "T__17", "T__18", "T__19", 
                  "T__20", "T__21", "T__22", "T__23", "T__24", "T__25", 
                  "T__26", "T__27", "SEMVER", "COMMENT", "WS", "LINE_COMMENT", 
                  "ESCAPED_QUOTE", "QUOTED_STRING", "MUL", "DIV", "ADD", 
                  "SUB", "LT", "GT", "LE", "GE", "EQ", "NE", "BOOL_LIT", 
                  "CNAME", "DATATYPE", "PREFLOAT", "FLOAT", "NAT", "EXP", 
                  "LETTER", "DIGIT", "METADATA" ]

    grammarFileName = "Relay.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


