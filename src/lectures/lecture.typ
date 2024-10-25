#import "/utils.typ": *

#set page("a5", numbering: "1")
#set text(lang: "ru")
#set par(justify: true)
#set heading(numbering: "1.")


`20 Сентября 2024`
= Линейные системы

== Постановка задачи

$ t in [0, T] arrow.r.bar.long x(t) = vec(x_1(t), dots.v, x_n (t)) $

#cbox[
Линейное неоднородное уравнение:
$ der(x)(t) = A(t) x(t) + f(t) when t in (0, T) wide (1) $
]


$ A(t) = ((a_(i j)(t))) when i, j = overline(1","n) $

$ f(t) = vec(f_1(t), dots.v, f_n (t)) $

$
cases(
  der(x)_1 = a_(11)x_1+a_(12)x_2+dots+a_(1 n)x_n +f_1,
  dots.v,
  der(x)_n = a_(n 1)x_1 + a_(n 2)x_2 + dots + a_(n n)x_n + f_n
)
$


#cbox[
  Линейное однородное уравнение:
  $ (1_0) wide der(x) = A(t)x $
]

#cbox[
  Задача Коши для (1):
  
  $ (2) wide x(0) = x^((0)) in RR^n $
]





== Матмодели

=== Температура в доме

$x_(1,2)(t)$ --- температура на 1, 2 этаже \
$x_g$ --- температура земли \
$x_e$ --- температура на улице

$ der(x)_1 = k_1 (x_g - x_1) + k_2(x_2 - x_1) + k_3(x_e - x_1) + p(t) $
Коэффициент передачи через пол, через потолок, через стены + печка.

$ der(x)_2 = k_2(x_1 - x_2) + k_4(x_e - x_2) $

Числа $k_(1,2,3,4)$ известны.

$ A = mat(
  -(k_1 + k_2 + k_3), k_2;
  k_2, -(k_2+k_4);
)
wide
f = vec(
  p+k_1 x_g + k_3 x_e,
  k_4 x_e
)
$



=== Динамика цен и запасов

$s(t)$ --- объём продаж за единицу времени.\
$p(t)$ --- текущая цена.\
$I(t)$ --- уровень запасов на каком-то складе.\
$Q(t)$ --- скорость поступления товара.\
$p_*$ --- равновесная цена.\
$I_*$ --- желаемый запас.

$
cases(
  der(s) = beta(p - p_*) when beta < 0,
  der(p) = alpha(I - I_*) when alpha < 0,
  der(I) = Q - s
)
wide
x(t) = vec(s(t), p(t), I(t))
$

$
A = mat(
  0, beta, 0;
  0, 0, alpha;
  -1, 0, 0;
)
wide
f = vec(-beta p_*, -alpha I_*, Q)
$




== Корректность задачи Коши

$ &(1) wide der(x)(t) = A(t)x(t) + f(t) when t in (0, T)\

  &(2) wide x(0) = x^((0)) \

  &(*) wide cases(
    t>0,
    x^((0)) in RR^n,
    f in C[0, T],
    A in C[0, T] space (a_(i j) in C[0, T])
  )
$

#bbox[Th.1][
Пусть выполнены условия $(*)$. Тогда $exists!$ решение (1) - (2).
]



== Априорные оценки решения задачи Коши

$ x,y in RR^n wide |x| = sqrt(sum_1^n x_j^2) $

$ (x, y) = sum_1^n x_j y_j $

$ (der(x), x) = (A x, x) + (f, x) $

С другой стороны:

$  1/2 dif / (dif t)  |x|^2 = (der(x), x) $


Неравенство Коши-Буняковского:

$ |(x, y)| <= |x| dot |y| $

$ 1/2 dif / (dif t) |x|^2 <= |A x| dot |x| + |f| dot |x| $


$ |underbrace(A x, y)| = sqrt(sum y_i^2) $

$ y_i = sum_(j=1)^n a_(i j)x_j <= sqrt(sum_(j=1)^n a_(i j)^2) dot |x| $

$ M_1 = max_(t in [0, T] \ i in [1, n]) sum_(j=1)^n a_(i j)^2 $

$ y_i^2 <= M_1 |x|^2 $

$ y = abs(A x) = sqrt(sum y_i^2) <= sqrt(M_1 n abs(x)^2) $

$ M_2 = max_(t in [0, T]) f(t) $

$ 1/2 d / (d t) |x|^2 <= sqrt(M_1 n) |x|^2 + M_2 |x| $

$ d / (d t) |x| <= sqrt(M_1 n) |x| + M_2 $

$ d / (d t) |x| - sqrt(M_1 n) |x| <= M_2 $

$ (d / (d t) |x| - sqrt(M_1 n) |x|)e^(-sqrt(M_1 n)t) <= M_2e^(-sqrt(M_1 n)t) $

$ d / (d t) (e^(-sqrt(M_1 n)t) |x(t)|) <= M_2e^(-sqrt(M_1 n)t) $

Интегрируем:

$ integral_0^t d / (d t) (e^(-sqrt(M_1 n)t) |x(t)|) <=
integral_0^t M_2e^(-sqrt(M_1 n)t) $
        

$ e^(-sqrt(M_1 n)t) (|x(t)| - |x^((0))| )<=
M_2 / (sqrt(M_1 n)) (1-e^(-sqrt(M_1 n)t)) $

$ cbox(|x(t)| <= e^(sqrt(M_1 n)t) |x^((0))| t + M_2/(sqrt(M_1 n)) (e^(sqrt(M_1 n) t) - 1)
) $

== Однородная система линейных ОДУ

$ (3) wide der(x) = A(t) x when t in (0, T) $




#note[
  Пусть
  $ cases(
      der(x) = A x when  t in (0, T),
      x(0) = 0
    )
  $
  Тогда, существует единственное решение $x(t) eq.triple 0$ 
]


#bbox[Лемма 1][
  Множество решений (3) есть линейное пространство
]


#def[
  Пусть вектор-функции $x^((1)), dots, x^((m)) in C[0, T]$


  $ x^((1)) = vec(x^((1))_1, dots.v, x^((1))_n), ... $
  

  Система векторов называется линейно независимой, если:
  
  $ sum_(j=1)^m c_j x^((j))(t) = 0 when t in [0, T] => c_1 = dots = c_m = 0 $
]

#def[
  Система из $n$ линейно независимых решений однородной задачи (3) называется
  фундаментальной системой решений.
]


#def[
  Пусть $x^((1)), ..., x^((n))$ --- решение (3)
  
  $W(t) = det(x^((1))(t), dots, x^((n))(t))$ --- определитель Вронского.
]

#def[
  $Phi(t) = (phi^((1)), dots, phi^((n)))$ --- фундаментальная матрица системы (3),
  где $phi^((1)), dots, phi^((n))$ --- ф.с.р.
]

#bbox[Лемма 2][
  $ det Phi(t) != 0 when t in [0, T] $
]

`27 Сентября 2024`

$ A(t) = ((a_(i j)(t)))_(i,j = overline(1 comma n)) wide n in NN, space n>=2 $

$ (1) quad der(x) = A(t) x
wide x = vec(x_1(t), dots.v, x_n (t))
wide t in [0, T] $

#note[
  $ (2) wide der(Phi)(t) = A(t) Phi(t) $
]

#statement[
  Пусть $B(t)$ дифференциируемая матрица, $det B(t) != 0$

  $ dif / (dif t) (det B(t)) = det (B(t)) dot tr (B^(-1) der(B)) $
]

Пусть $Phi$ --- фундаментальная матрица.

$ dif / (dif t) det Phi(t) = det(Phi(t)) tr(Phi^(-1) der(Phi)) =
det(Phi(t)) tr(Phi^(-1) A Phi)
$
Так как
$W(t) = det Phi(t)$, а след матрицы обладает свойством $tr(A B) = tr(B A)$
, получаем:
#cbox[
$ der(W) = W tr A $
]


Формула Остроградского-Лиувилля:
#cbox[
$ W(t) = W(t_0) exp(integral_(t_0)^(t) tr A(s) dif s) $
]

#bbox(breakable: true)[Теорема][
  Пусть $Phi(t), t in [0, T] -$ фундаментальная матрица (1).
  
  Тогда, $x(t), t in [0, T] -$ решение (1) $<=> cases(
    x(t) = Phi(t) c,
    c = vec(c_1, dots.v, c_n) = "const"
  )$
][
$(arrow.double.l)$:\
$ der(x) = der(Phi) c = A Phi c = A dot Phi c space $
Значит, $Phi c -$ решение. $qed$
][
$(=>)$:

Пусть $ x = x(t) -$ решение (1).

Рассмотрим СЛАУ $Phi(0) c = x(0)$

$det Phi(t) != 0$

$exists c in RR^n$

$y(t) := Phi(t) c - $ решение (1)

Покажем, что $y equiv x$

$der(x) = A x wide  der(y) = A y wide x(0) = y(0) $

В силу единственности решения задачи Коши:

$x(t) = y(t) = A(t) c space qed$
]

#note[
  Общее решение (1):
  $ (3) wide x(t) = Phi(t) c where c in RR^n $
]

#bbox[
  Теорема
][
  Существует фундаментальная система решений для системы (1).
][
  Ищем решение матричного дифференциального уравнения:
  $ der(Phi) = A Phi,  Phi(0) = I $
  Пусть $phi -$ первый столбец
  
  $der(phi) = A phi, phi(0) = vec(1, 0, dots.v, 0), exists!$ решение задачи Коши.

  $det Phi(0) = 1 != 0 => det Phi(t) != 0 when t in [0, T]$
]

#bbox[Замечание 2][
  Пусть $X-$ пространство решений (1). Из формулы (3) следует, что $dim X = n$
]

#eg[
  $
    cases(
      der(x)_1 = x_2,
      der(x)_2 = -x_1,
    ) wide
    A = mat(0, 1; -1, 0;)
  $

  $
    der(x) = A x
    wide phi^((1))(t) = mat(sin t; cos t)
    wide phi^((2))(t) = mat(-cos t; sin t)
  $

  $
  det Phi(t) = det mat(sin t, -cos t; cos t, sin t) = 1!= 0 $

  $ x(t) =  mat(sin t, -cos t; cos t, sin t) vec(c_1, c_2) $

  $ x_1 = c_1 sin t - c_2 cos t \
    x_2 = c_1 cos t + c_2 sin t
  $
]

== Система линейных неоднородных диффуров

$ (1) wide der(x) = A(t) x + f(t) where f(t) = vec(f_1(t), dots.v, f_n (t)) equiv.not 0 $

#bbox[Теорема][
  Пусть $Phi(t) where t in [0, T] -$ фундаментальная система однородной системы.

  $x = hat(x)(t) where t in [0, T]-$ частное решение (1)

  $x = x(t) -$ решение (1) $<=> x(t) = Phi(t) с + hat(x)(t) quad (2)$ 

  То есть, о.р.н.с. = о.р.о.с. + ч.р.н.с.
][
  $(arrow.double.l)$:

  Обозначим $y(t) := Phi(t) с$

  $der(y) = A y$

  $der(x) = der(y) + der(hat(x)) 
  = A y + A hat(x) + f 
  = A (y + hat(x)) + f
  = A x + f
  space qed$
][
  $(=>)$:

  $x-$ решение (1)

  $y:=x - hat(x)$

  $der(y) = der(x) - der(hat(x)) = A x + f - A hat(x) - f = A y => y = Phi c space qed$
]

=== Метод вариации произвольных постоянных

$x(t) = Phi(t) c(t)$

Подставим в (1)

$der(Phi) c + Phi der(c) = A Phi c + f$

Так как, $der(Phi) = A Phi$, то:

$A Phi c + Phi der(c) = A Phi c + f$

$Phi der(c) = f$

#lbox[
  $ der(c) = Phi^(-1) f $
]

$ c(t) = integral_0^t Phi^(-1) (s) f(s) dif s + K where K - " настоящая константа" $

#eg[
  $ cases(
    der(x)_1 = x_2 + 1,
    der(x)_2 = -x_1,
  ) $

  $ Phi = mat(sin t, -cos t; cos t, sin t) $

  $ 
    cases(
        sin der(c)_1 - cos der(c)_2 = 1,
      cos der(c)_1 + sin der(c)_2 = 0
    )
  $

  $ der(c)_1 = det mat(1, -cos t; 0, sin t) = sin t \
    der(c)_2 = det mat(sin t, 1; cos t, 0) = - cos t $

  $ c_1 = -cos t + K_1 \
    c_2 = -sin t + K_2
  $

  $ x = (-cos t + K_1) vec(sin t, cos t) + (-sin t + K_2) vec(-cos t, sin t) $

  $ x = K_1 vec(sin t, cos t) + K_2 vec(-cos t, sin t) + vec(0, -1) $
]

=== Решение задачи Коши

$ (C P) cases(
  der(x) = A(t) x + f(t),
  x(0) = x^((0)) in RR^n
) $

$ x(t) = Phi(t) (integral_0^t Phi^(-1)(s) f(s) dif s + K) $

$ x^((0)) = Phi(0)K, space K = Phi^(-1)(0)x^((0)) $

#cbox[
$ x(t) = Phi(t)  Phi^(-1)(0) x^((0)) + integral_0^t Phi(t) Phi^(-1)(s) f(s) dif s  $
]

#def[
  Матрица Коши (импульсная матрица):
  $ K(t, s) = Phi(t) Phi^(-1)(s) $
]

#cbox[
$ x(t) = K(t, 0) x^((0)) + integral_0^t K(t, s) f(s) dif s $
]

`4 Октября 2024`

#bbox[
  Теорема повышения гладкости
][
  $ x = x(t) where t in [0, T] - "решение" $
  $ der(x) = A(t) x + f(t) where 0 < t < T   $
  $ A in C^k [0, T] where f in C^k [0, T] $
  $ k = 0, 1, dots $
  Тогда,
  $ x in C^(k+1)[0, T] $
][
  $ der2(x) = der(A) x + A der(x) + der(x) $ 
  $ der(A) x + A der(x) + der(x) in C[0, T] $
  $ => x in C^2 [0, T] "и т.д. " qed $
]

= Системы диффуров с постоянными коэффициентами

Однородная система:

$ (1) quad der(x) = A x, t in [0, T] $
$ A = ((a_(k j)))_(k, j = overline(1 comma n)) in RR^(n times n) $
$ x = vec(x_1(t), dots.v, x_n (t)) $


#note[
  Рассмотрим случай, когда Наташа равна единичке.
  Если $ der(x) = a x$, тогда $ x = C e^(a t)$
]

== Матричная экспонента

$ "Пусть" A in RR^(n times n), A = ((a_(k j))) $
$ norm(A) = max_(1 <= k <= n) sum_(j=1)^n abs(a_(k j)) $
$ A_m -->_(m -> oo) A quad eq.def quad norm(A_m - A) -->_(m -> oo) 0 $

$ B = sum_(m=1)^oo A_m quad eq.def quad norm(B - sum_(m=1)^N A_m) ->_(N->oo) 0 $

#def[
  $ exp A = e^A = sum_(j=0)^oo 1/j! A^j $
]

#bbox[Лемма][
  $ forall A in RR^(n times n) "матричный ряд" e^A "сходится" $
][
  $ S_m = sum_(j=0)^m 1/j! A^j $
  $ "Пользуясь фактом" norm(A^j) <= norm(A)^j $
  $ norm(S_m - S_(m+k)) = norm(sum_(j=m+1)^(m+k) 1/j! A^j) <= sum_(j=m+1)^(m+k) 1/j! norm(A)^j -->_(m->oo) 0 space qed$
]

#note[
  $ e^(A + B) = e^A e^B = e^B e^A <=> A B = B A$
]

#bbox[Теорема][
  $ Phi(t) = e^(t A) - "фундаментальная матрица системы (1)" $
][
  $ der(Phi)(t) = lim_(h->0) 1/h (e^((t+h)A) -e^(t A)) = lim_(h->0) 1/h (e^(h A) - I) e^(t A) =\
  = lim_(h->0) 1/h (I + h A + 1/2! (h A)^2 + dots - I)e^(t A) = A e^(t A)
  $

  $ det Phi(0) = 1 != 0 => qed $
]

#bbox[Следствие 1][
  $ "Общее решение (1):"\ x(t) = e^(t A)c where c = vec(c_1, dots.v, c_n) $
]

#bbox[Следствие 2][
  $ "Решение задачи Коши: "\
  x=A x, space   x(t_0) = x^((0))\ x(t) = e^((t-t_0)A)  x^((0)) $
]

== Структура $e^(t A) $

=== Жорданова форма матрицы

#def[
  $ "Функция" lambda |-> det(P - lambda I) =\
  = (-1)^n lambda^n + a_1 lambda^(n-1) + dots + a_(n-1) lambda + a_n \
  "это характеристический многочлен матрицы" P
  $
]

$ a_1 = (-1)^n tr(P) $
$ a_n = det(P) $

#def[
  $ P, Q in RR^(n,n) "называются подобными" (P tilde Q)", если"\
  exists S, det S != 0\
  Q = S^(-1) P S "или" S Q = P S\
  $
]

$ det( Q- lambda I) = det(S^(-1) ( P - lambda I)S) = det(P - lambda I) $
У подобных матриц одинаковы и следы и определители.

#def[
  $ J = "diag"{J_1, J_2, dots, J_k}  $
  
  #figure(image("image.png", width: 50%))
  $ J_m = mat(
    lambda_m, 1;
    ,lambda_m,1;
    ,,dots.down, dots.down;
    ,,,lambda_m, 1;
    ,,,,lambda_m
  ) - " клетка Жордана" $
]

#bbox(breakable: true)[
  Теорема
][
  $ A tilde J = "diag"{J_0, J_1, dots, J_q} $
  $ J_0 = mat(
    lambda_1;
    ,lambda_2;
    ,,dots.down;
    ,,,lambda_p
  ) 
  quad
    J_k = mat(
    lambda_(p+k), 1;
    ,lambda_(p+k), 1;
    ,,dots.down, dots.down;
    ,,,lambda_(p+k), 1;
    ,,,,lambda_(p+k);
  ) $

  где $lambda_1, dots, lambda_p - $ простые собственные числа $A$
  
  $lambda_(p+k) -$ кратные собственные   числа $A$ кратности $r_k$

  $ n = p+ q $

Даже если матрица имеет элементы $ in RR$, её собственные числа могут  $in CC$.

$ A = S J S^(-1), det S != 0 $
$ A^k = S J underbracket(S^(-1) dot S, I) J S^(-1) dot dots dot S J S^(-1) = S J^k S^(-1) $
$ e^(t A) = sum_(j=0)^oo 1/j! (t A^j) = sum_(j=0)^oo 1/j! t^j S J^j S^(-1) = S e^(t J) S^(-1) $
$ Psi(t) := e^(t J) = S^(-1) e^(t A) S $
$ der(Psi) =  S^(-1) A e^(t A) S $ 
]

=== Экспонента Жордановской матрицы

#bbox[Утверждение 1][
  $ J = "diag"{J_0, J_1, dots, J_q} => $
  $ e^(t J) = "diag"{e^(t J_0), e^(t J_1), dots, e^(t J_q)} $
]

#bbox[Утверждение 2][
  $ e^(t J_0) = mat(
    e^(t lambda_1);
    ,e^(t lambda_2);
    ,,dots.down;
    ,,,e^(t lambda_p)
  ) $
]

$ B = lambda I + H,

  quad

  H = mat(
      0, 1;
    , 0, 1;
    ,,0,1;
    ,,,dots.down,dots.down;
    ,,,,0,1;
    ,,,,,0
) $

#let etb = $ e^(lambda t) mat(
    1, t, 1/2!t, dots, 1/(r-1)! t^(r-1);
    , 1, t, dots, 1/(r-2)! t^(r-2);
    ,,dots.down,dots.down;
    ,,,1,t;
    ,,,,1
  ) $


#bbox[Утверждение 3][
  $ "Пусть" B = mat(
    lambda, 1;
    , lambda, 1;
    ,,lambda,1;
    ,,,dots.down,dots.down;
    ,,,,lambda,1;
    ,,,,,lambda
  ) space r "столбцов"  $

  $ e^(t B) = e^(t lambda I) e^(t H) = e^(lambda t ) I e^(t H) = e^(lambda t) e^(t H) $

  $ e^(t H) = I + t H + 1/2! t^2 H^2 + dots $

  $ H^2 = mat(
    0, 0, 1, 0, 0, dots, 0;
    0, 0, 0, 1, 0, dots, 0;
    0, 0, 0, 0, 1, dots, 0;

    ,,,,,dots.down;
    0, 0, 0, 0, 0, dots, 1;
    0, 0, 0, 0, 0, dots, 0
  ) $
  
   $ e^(t B) = etb $
]

#bbox[Теорема][
  $ J = "diag"{J_0, J_1, dots, J_q} $
  $ "Тогда" e^(t J) = \ 
  = mat(
  mat(
    e^(t lambda_1),,;
    ,dots.down,;
    ,,e^(t lambda_p)
  ),;
  ,etb;
  ,,dots.down
) $
  
]

`18 Октября 2024`

$ Phi(t) = e^(t A) = S "diag"{e^(t J_0), e^(t J_1), dots, e^(t J_q)} S^(-1) $

$ e^(t J_k) = e^(t lambda_k) mat(
  1, t, 1/2! t^2, dots, t^(r_k - 1)/(r_k - 1)!;
  ,1,t,dots.down;
  ,,1,dots.down,;
  ,,,dots.down,t;
  ,,,,1
) $

$ r_k "кратность собственных значений" lambda_k $

=== Метод Эйлера

Пусть $lambda$ собственное число $A$ кратности $r$.

$lambda$ соответствует решение (1)

#cbox[
$ x(t) = e^(lambda t) Q(t), quad Q - "многочлен, степени" <= r - 1 $ 
]

/1/

#bbbox[Пример 1][

$ cases(
    der(x_1) = 4x_1 - x_2,
    der(x_2) = 5x_1 + 2 x_2
  ) $

$
  A = mat(
    4, -1; 5, 2
  )
$

$ matdet(4-lambda, -1; 5, 2-lambda) = 0 <=> lambda = 3 plus.minus 2i $

$ phi(t) = mat(a;b) e^((3+2i)t) $

$ (3+ 2i) vec(a, b) cancel(e^dots) = vec(4a - b, 5a + 2b) cancel(e^...) = A phi $

$ 
  a = 1, b = 1 - 2i
$

$ phi(t) = vec(1, 1-2i) e^((3+2i)t) = vec(1, 1-2i) e^(3t) (cos(2t) + i sin(2t)) $

$ x(t) = c_1 e^(3 t) vec(cos 2 t, cos 2t + 2 sin 2t) + c_2 e^(3 t) vec(sin 2t, -2cos 2 t + sin t)  $
]

#bbbox[Пример 2][
  $ der(x) = A x, space A = mat(
    2, 1, 1;
    -2, 0, -1;
    2, 1, 2
  )
  // der(x)_1 = 2x_1 + x_2 + x_3 
  
  $

  $ (lambda - 2)^1(lambda - 1)^2 = 0 $

  + $lambda=2$
    $ phi(t) = vec(a, b, c) e^(2 t) $
    $ 2 vec(a, b, c)  = vec(2a + b + c, -2a - c, 2a + b + 2c) $

    $a = 1, c=2, b = -2 $

    $phi(t) = vec(1, -2, 2) e^(2 t)$

  + $lambda =1, space r = 2$
    $ phi(t) = vec(
      alpha_1 t + alpha_2,
      beta_1 t + beta_2,
      gamma_1 t + gamma_2
    ) e^t $

    Первая строчка:

    $ (alpha_1 + (alpha_1 t + alpha_2)) e^t = (2(alpha_1t + alpha_2) + beta_1t + beta_2 + gamma_1 t + gamma_2) e^(t) $

    $ cases(
        alpha_1 + alpha_2 = 2 alpha_2 + beta_2 + gamma_2,
        alpha_1  = 2 alpha_1 + beta_1 + gamma_1        
    ) $
    Вторая строчка:
    
    $ beta_1 + beta_1 t + beta_2 = -2(alpha_1 t + alpha_2) - gamma_1 t - gamma_2$

    Третья аналогично:

    $ gamma_1 + gamma_1 t + gamma_2 = 2(alpha_1 t + alpha_2 + beta_1 t + beta_2 + 2(gamma_1 t + gamma_2))$

    Решаем алгебру:

    $ alpha_1 = 0, space
      beta_1 = - gamma_1, space
      beta_1 = - alpha_2, space
      beta_2 = beta_1 - gamma_2$

    $ cases(alpha_1 = 0, space
     alpha_2 = c_2, space
     beta_1 = -c_2, space
     beta_2 = -c_2 - c_3, space
     gamma_1 = c_1, space
     gamma_2 = c_3) $
    
  $ "Ответ:" x = c_1 e^(2 t) vec(1, -2, 2) + vec(c_2, -c_2 t - (c_2 + c_3), c_2 t + c_3) e^t = \
  = c_1 e^(2 t) vec(1, -2, 2) + c_2 e^t vec(1, -t -1, t) + c_3 e^t vec(0, -1, 1)
  $
    
]

== Неоднородные системы

$ (1_n) quad der(x) = A x + f(t) $
$ A in RR^(n times n) $
$ x(t) = hat(x)(t) + Phi(t) c $
$ c in RR^n, space hat(x) - "частное решение", space Phi(t)c - "о.р.о.с." $

#note[
  $ der(x) = A x + k f_1(t) + f_2(t), space k = "const" $
  
  $ der(x)^((1)) = A x^((1)) + f_1(t) $

  $ der(x)^((2)) = A x^((2)) + f_2(t) $

  $ => x = k x^((1)) + x^((2)) $
]

Неоднородность в виде квазимногочлена:

$ f(t) = e^(mu t) P(t), space deg P = m, space P(t) = vec(P_1(t), dots.v, P_n (t)) $

$ mu in CC $

#note[
  $ "Пусть" f(t) = sin 3 t vec(t^2, t) $
  $ tilde e^(3 i t) vec(t^2, t) $
]

=== Нерезонансный случай

$ mu "не является собственным значением" A$

$ der(x) = A x + e^(mu t) P(t)$

$ hat(x)(t) = e^(mu t) Q(t), space deg Q <= m, space Q(t) = sum_(j=0)^m q_j t^j, space q_j in RR^n$

$ (mu Q + der(Q)) cancel(e^(mu t)) = A e^(mu t) Q + cancel(e^(mu t)) P$

#lbox[
$ (mu I - A)Q = P - der(Q) $
]

$ P(t) = sum_(j=0)^m p_j t^j$

$ t^m: (mu I = A) q_m = p_m$

$ q_m = (mu I - A)^(-1) p_m$

$ t^(m-1): (mu I - A) q_(m-1) = p_(m-1) - m q_m = p_(m-1) - m (mu I - A)^(-1) p_m$

=== Резонансный случай
`25 Октября 2024`


$ exists lambda_k - "собственное значение" A, space mu = lambda_k $

$ hat(x) (t) = e^(mu t) Q(t)  $

$ &deg Q <= m + l, space l - "максимальный размер Жордановской клетки,"\
&"соответствуеющей" lambda_k $

$ l <= "кратности собственного значения" $

#bbbox[Пример 1][
  $ cases(
      der(x)_1 = 2x_1 + x_2 + e^(2 t),
      der(x)_2 = 3x_1 + 4x_2
  ) $

  $ A = mat(2, 1; 3, 4) $

  $ matdet(2-lambda, 1; 3, 4-lambda) = 0 wide lambda_1 = 1, space lambda_2 = 5 $

  $ f(t) = e^(2 t) vec(1, 0), space m=0 $

  $ hat(x) (t) = e^(2 t) vec(p, q) $

  Подставляем в задачу и получаем:
  
  $ q = -1, space p = 2 / 3 $

  $ hat(x) = e^(2 t) vec(2/3, -1) $

  $ A vec(a, b) = 1 vec(a, b) $

  $ cases(
      2a + b = a,
      3a + 4b = b
  )
  $

  $ vec(1, -1) - "собственный вектор, соответсветсвтуеющий "  lambda = 1 $

  $ vec(1, 3) quad tilde quad lambda = 5 $

  $ x(t) = c_1 e^t vec(1, -1) + c_2 e^(5 t) vec(1,3) + e^(2 t) vec(2/3, -1) $

  Задача тов. Коши:

  $ x(0) = 0 $

  $ c_1 vec(1, -1) + c_2 vec(1, 3) + vec(2/3, -1) = 0 $
  
  $ c_1 = -3/4, space c_2 = 1/12  $

  Ответ:
  
  $ cases(
    x_1 = -3/4 e^t + 1/12 e^(5 t) + 2/3 e^(2 t),
    x_2 = 3/4 e^t + 1/4 e^(5 t) - e^(2 t)
  ) $
]

#bbbox[Пример 2][
  $ cases(
      der(x)_1 = x_2 + epsilon sin t,
      der(x)_2 = -x_1
  ), quad t>0 $

  $ x_1 (0) = x_2 (0) = 0 $

  Можно решать при $epsilon = 1$, а потом домножать ответ.

  $ sin t = "Im" e^(i t) $

  $ der(x) = mat(0, 1; -1, 0) x + e^(i t) vec(1, 0) $

  $ matdet( -lambda, 1; -1, -lambda) = 0 <=> lambda^2 +1 =0 <=> lambda = plus.minus i $

  $ A vec(a, b) = i vec(a, b) $

  $ b = i a => vec(1, i) $

  $ x^((0)) = c_1  e^(i t) vec(1,i) + c_2 e^(-i t) vec(1, -i) $

  $ hat(x) (t) = e^(i t) vec(a_1 t + b_1, a_2 t + b_2) $

  Теперь подставляем в задачу и находим четыре числа:

  $ i e^(i t) vec(a_1 t + b_1, a_2 t + b_2) + e^(i t) vec(a_1, a_2) = e^(i t) vec(a_2 t + b_2, -a_1 t - b_1) + e^(i t) vec(1, 0) $

  $ i  vec(a_1 t + b_1, a_2 t + b_2) +  vec(a_1, a_2) =  vec(a_2 t + b_2, -a_1 t - b_1) + vec(1, 0) $

  $ t^1: space i a_1  = a_2, space i a_2 = -a_1 $

  $ t^0: space i b_1 + a_1 = b_2 + 1, space i b_2 + a_2 = -b_1 $

  $ a_1 = 1/2, space a_2 = i/2, space i b_1 = b_2 + 1/2  $

  $ a_1 = 1/2, space a_2 = i/2, space b_1 = 0, space b_2 = -1/2 $

  $ der(x) (t) = e^(i t) vec(1/2 t, i/2 t - 1/2) $  

  $ x(t) = e^(i t) vec(1/2 t, i/2 t - 1/2) + c_1 e^(i t) vec(1, i) + c_2 e^(-i t) vec(1, -i) $

  $ x(0) = 0 <=> cases(
    c_1 + c_2 = 0,
    -1/2 + c_1 i -c_2 i = 0
  ) $

  $ c_1 = -1/4 i, space c_2 = 1/4 i $

  $ x(t) = e^(i t) vec(1/2 t, i/2 t - 1/2) - 1/4 i vec(1, i) e^(i t) + 1/4 i e^(-i t) vec(1, -i) $

  $ "Im" x(t) = vec(
      1/2 t sin t - 1/4 cos t + 1/4 cos t,
      1/2 t cos t - 1/2 sin t + 1/4 sin t - 1/4 sin t
    ) $

  Ответ:

  $ x_1 (t)  = epsilon / 2 t sin t $

  $ x_2 (t) = epsilon / 2 ( t cos t - sin t) $
    
]

= Устойчивость решений систем обыкновенных дифференциальных уравнений

== Определение устойчивости и простейшее применение

#cbox["Fur fier kein beer"]

#cbox[
$ (1) quad der(x) (t) = f(t, x(t)), space t >0 $

$ x(t) = vec(x_1 (t), dots.v, x_n (t)) $
]

#def[
  Пусть $x = phi(t), space t >= 0$ --- решение (1).
  Решение $phi(x)$ называется устойчивым по Ляпунову, если
  $ forall epsilon > 0 and forall "решения" x = psi(t) "системы (1)," $
  $ exists delta_epsilon > 0 $
  $ "такого, что" abs(phi(0) - psi(0)) < delta_epsilon $
  $ => mod(phi(t) - psi(t)) < epsilon $
  $ forall t > 0 $
]

#def[
  Решение $x = phi(t)$ называется асимптотически устойчивым,
  если к дополнению к этому,
  $ lim_(t->oo) abs(phi(t) - psi(t)) = 0 $
]
