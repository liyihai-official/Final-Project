% mypresentation.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{my-Presentation}[2024/09/14 Custom Beamer Class]

% 基于 beamer 类
\LoadClass[aspectratio=3218]{beamer}

% 设置目录深度
\setcounter{tocdepth}{2}

% 在每个章节开始时插入目录
\AtBeginSection[]
{
  \begin{frame}<beamer>
    \frametitle{Outline}
    \begin{multicols}{2}
      \tableofcontents[currentsection]
    \end{multicols}
  \end{frame}
}

% 设置字体大小
\setbeamerfont{footnote}{size=\tiny}
% \setbeamerfont{frametitle}{size=\huge} % 如需修改 frametitle 字体大小，取消注释
\setbeamerfont{framesubtitle}{size=\large}

% 设置主题和字体
\RequirePackage{mathptmx}
\RequirePackage{amsmath}
\RequirePackage{mathspec}
\usefonttheme{serif}
\setmainfont{Times New Roman}
\setmathsfont(Digits,Latin){Times New Roman}
\usetheme{Madrid}

% 加载必要的宏包
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption, wrapfig, color}
\RequirePackage{hyperref}
\RequirePackage{booktabs}
\RequirePackage{tikz}
\usetikzlibrary{positioning, arrows.meta}
\RequirePackage{booktabs, multirow}
\RequirePackage{multicol}
% 设置参考文献样式
\RequirePackage[style=numeric-comp,backend=biber]{biblatex}
% 注意：\addbibresource 需要在主文档中使用