??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
s
mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namemean/kernel
l
mean/kernel/Read/ReadVariableOpReadVariableOpmean/kernel*
_output_shapes
:	?*
dtype0
j
	mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	mean/bias
c
mean/bias/Read/ReadVariableOpReadVariableOp	mean/bias*
_output_shapes
:*
dtype0
?
log_std_dev/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namelog_std_dev/kernel
z
&log_std_dev/kernel/Read/ReadVariableOpReadVariableOplog_std_dev/kernel*
_output_shapes
:	?*
dtype0
x
log_std_dev/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namelog_std_dev/bias
q
$log_std_dev/bias/Read/ReadVariableOpReadVariableOplog_std_dev/bias*
_output_shapes
:*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:?*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
	_body
_mu
_log_std
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
16
17
18
19
f
0
1
"2
#3
$4
%5
(6
)7
*8
+9
10
11
12
13
?

.layers
regularization_losses
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
1metrics
2non_trainable_variables
 
?
3axis
	gamma
beta
 moving_mean
!moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
h

"kernel
#bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?
<axis
	$gamma
%beta
&moving_mean
'moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

(kernel
)bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
?
Eaxis
	*gamma
+beta
,moving_mean
-moving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
 
v
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
F
0
1
"2
#3
$4
%5
(6
)7
*8
+9
?

Jlayers
regularization_losses
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
Mmetrics
Nnon_trainable_variables
FD
VARIABLE_VALUEmean/kernel%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUE	mean/bias#_mu/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Olayers
regularization_losses
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
Rmetrics
Snon_trainable_variables
RP
VARIABLE_VALUElog_std_dev/kernel*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUElog_std_dev/bias(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Tlayers
regularization_losses
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
Wmetrics
Xnon_trainable_variables
US
VARIABLE_VALUEbatch_normalization/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_1/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_1/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 
*
 0
!1
&2
'3
,4
-5
 
 

0
1
 2
!3

0
1
?

Ylayers
4regularization_losses
Zlayer_regularization_losses
[layer_metrics
5	variables
6trainable_variables
\metrics
]non_trainable_variables
 

"0
#1

"0
#1
?

^layers
8regularization_losses
_layer_regularization_losses
`layer_metrics
9	variables
:trainable_variables
ametrics
bnon_trainable_variables
 
 

$0
%1
&2
'3

$0
%1
?

clayers
=regularization_losses
dlayer_regularization_losses
elayer_metrics
>	variables
?trainable_variables
fmetrics
gnon_trainable_variables
 

(0
)1

(0
)1
?

hlayers
Aregularization_losses
ilayer_regularization_losses
jlayer_metrics
B	variables
Ctrainable_variables
kmetrics
lnon_trainable_variables
 
 

*0
+1
,2
-3

*0
+1
?

mlayers
Fregularization_losses
nlayer_regularization_losses
olayer_metrics
G	variables
Htrainable_variables
pmetrics
qnon_trainable_variables
#
	0

1
2
3
4
 
 
 
*
 0
!1
&2
'3
,4
-5
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 0
!1
 
 
 
 
 
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 

,0
-1
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammadense/kernel
dense/bias!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_1/betabatch_normalization_1/gammadense_1/kerneldense_1/bias!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_2/betabatch_normalization_2/gammamean/kernel	mean/biaslog_std_dev/kernellog_std_dev/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_34583885
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/kernel/Read/ReadVariableOpmean/bias/Read/ReadVariableOp&log_std_dev/kernel/Read/ReadVariableOp$log_std_dev/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_34585139
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemean/kernel	mean/biaslog_std_dev/kernellog_std_dev/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_34585209??
?1
?	
!__inference__traced_save_34585139
file_prefix*
&savev2_mean_kernel_read_readvariableop(
$savev2_mean_bias_read_readvariableop1
-savev2_log_std_dev_kernel_read_readvariableop/
+savev2_log_std_dev_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_mean_kernel_read_readvariableop$savev2_mean_bias_read_readvariableop-savev2_log_std_dev_kernel_read_readvariableop+savev2_log_std_dev_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::	?::::::	?:?:?:?:?:?:
??:?:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%	!

_output_shapes
:	?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: 
?
?
8__inference_batch_normalization_2_layer_call_fn_34585042

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829002
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584829

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_mean_layer_call_fn_34584756

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_345834462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference__wrapped_model_34582552
input_1'
continuous_actor_34582508:'
continuous_actor_34582510:'
continuous_actor_34582512:'
continuous_actor_34582514:,
continuous_actor_34582516:	?(
continuous_actor_34582518:	?(
continuous_actor_34582520:	?(
continuous_actor_34582522:	?(
continuous_actor_34582524:	?(
continuous_actor_34582526:	?-
continuous_actor_34582528:
??(
continuous_actor_34582530:	?(
continuous_actor_34582532:	?(
continuous_actor_34582534:	?(
continuous_actor_34582536:	?(
continuous_actor_34582538:	?,
continuous_actor_34582540:	?'
continuous_actor_34582542:,
continuous_actor_34582544:	?'
continuous_actor_34582546:
identity

identity_1??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallinput_1continuous_actor_34582508continuous_actor_34582510continuous_actor_34582512continuous_actor_34582514continuous_actor_34582516continuous_actor_34582518continuous_actor_34582520continuous_actor_34582522continuous_actor_34582524continuous_actor_34582526continuous_actor_34582528continuous_actor_34582530continuous_actor_34582532continuous_actor_34582534continuous_actor_34582536continuous_actor_34582538continuous_actor_34582540continuous_actor_34582542continuous_actor_34582544continuous_actor_34582546* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_56000752*
(continuous_actor/StatefulPartitionedCall?
IdentityIdentity1continuous_actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity1continuous_actor/StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1y
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
8__inference_batch_normalization_1_layer_call_fn_34584955

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827982
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_34583355
batch_normalization_input*
batch_normalization_34583317:*
batch_normalization_34583319:*
batch_normalization_34583321:*
batch_normalization_34583323:!
dense_34583326:	?
dense_34583328:	?-
batch_normalization_1_34583331:	?-
batch_normalization_1_34583333:	?-
batch_normalization_1_34583335:	?-
batch_normalization_1_34583337:	?$
dense_1_34583340:
??
dense_1_34583342:	?-
batch_normalization_2_34583345:	?-
batch_normalization_2_34583347:	?-
batch_normalization_2_34583349:	?-
batch_normalization_2_34583351:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_34583317batch_normalization_34583319batch_normalization_34583321batch_normalization_34583323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345825762-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_34583326dense_34583328*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_345830652
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_34583331batch_normalization_1_34583333batch_normalization_1_34583335batch_normalization_1_34583337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827382/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_34583340dense_1_34583342*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_345830912!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_34583345batch_normalization_2_34583347batch_normalization_2_34583349batch_normalization_2_34583351*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829002/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_namebatch_normalization_input
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_34583396
batch_normalization_input*
batch_normalization_34583358:*
batch_normalization_34583360:*
batch_normalization_34583362:*
batch_normalization_34583364:!
dense_34583367:	?
dense_34583369:	?-
batch_normalization_1_34583372:	?-
batch_normalization_1_34583374:	?-
batch_normalization_1_34583376:	?-
batch_normalization_1_34583378:	?$
dense_1_34583381:
??
dense_1_34583383:	?-
batch_normalization_2_34583386:	?-
batch_normalization_2_34583388:	?-
batch_normalization_2_34583390:	?-
batch_normalization_2_34583392:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_34583358batch_normalization_34583360batch_normalization_34583362batch_normalization_34583364*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345826362-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_34583367dense_34583369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_345830652
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_34583372batch_normalization_1_34583374batch_normalization_1_34583376batch_normalization_1_34583378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827982/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_34583381dense_1_34583383*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_345830912!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_34583386batch_normalization_2_34583388batch_normalization_2_34583390batch_normalization_2_34583392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829602/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_namebatch_normalization_input
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34582576

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584795

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?W
?
$__inference__traced_restore_34585209
file_prefix/
assignvariableop_mean_kernel:	?*
assignvariableop_1_mean_bias:8
%assignvariableop_2_log_std_dev_kernel:	?1
#assignvariableop_3_log_std_dev_bias::
,assignvariableop_4_batch_normalization_gamma:9
+assignvariableop_5_batch_normalization_beta:@
2assignvariableop_6_batch_normalization_moving_mean:D
6assignvariableop_7_batch_normalization_moving_variance:2
assignvariableop_8_dense_kernel:	?,
assignvariableop_9_dense_bias:	?>
/assignvariableop_10_batch_normalization_1_gamma:	?=
.assignvariableop_11_batch_normalization_1_beta:	?D
5assignvariableop_12_batch_normalization_1_moving_mean:	?H
9assignvariableop_13_batch_normalization_1_moving_variance:	?6
"assignvariableop_14_dense_1_kernel:
??/
 assignvariableop_15_dense_1_bias:	?>
/assignvariableop_16_batch_normalization_2_gamma:	?=
.assignvariableop_17_batch_normalization_2_beta:	?D
5assignvariableop_18_batch_normalization_2_moving_mean:	?H
9assignvariableop_19_batch_normalization_2_moving_variance:	?
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB#_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB*_log_std/kernel/.ATTRIBUTES/VARIABLE_VALUEB(_log_std/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_mean_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_mean_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_log_std_dev_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_log_std_dev_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20f
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_21?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
؎
?
__inference_call_5602650

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	?2
clip_by_valueh
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	?2

Identityh

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes
:	?2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	?: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34583474

inputs!
sequential_34583403:!
sequential_34583405:!
sequential_34583407:!
sequential_34583409:&
sequential_34583411:	?"
sequential_34583413:	?"
sequential_34583415:	?"
sequential_34583417:	?"
sequential_34583419:	?"
sequential_34583421:	?'
sequential_34583423:
??"
sequential_34583425:	?"
sequential_34583427:	?"
sequential_34583429:	?"
sequential_34583431:	?"
sequential_34583433:	? 
mean_34583447:	?
mean_34583449:'
log_std_dev_34583463:	?"
log_std_dev_34583465:
identity

identity_1??#log_std_dev/StatefulPartitionedCall?mean/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_34583403sequential_34583405sequential_34583407sequential_34583409sequential_34583411sequential_34583413sequential_34583415sequential_34583417sequential_34583419sequential_34583421sequential_34583423sequential_34583425sequential_34583427sequential_34583429sequential_34583431sequential_34583433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345831072$
"sequential/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0mean_34583447mean_34583449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_345834462
mean/StatefulPartitionedCall?
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0log_std_dev_34583463log_std_dev_34583465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_345834622%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value?
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_34584866

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34582738

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_34584842

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345825762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_34583242

inputs*
batch_normalization_34583204:*
batch_normalization_34583206:*
batch_normalization_34583208:*
batch_normalization_34583210:!
dense_34583213:	?
dense_34583215:	?-
batch_normalization_1_34583218:	?-
batch_normalization_1_34583220:	?-
batch_normalization_1_34583222:	?-
batch_normalization_1_34583224:	?$
dense_1_34583227:
??
dense_1_34583229:	?-
batch_normalization_2_34583232:	?-
batch_normalization_2_34583234:	?-
batch_normalization_2_34583236:	?-
batch_normalization_2_34583238:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_34583204batch_normalization_34583206batch_normalization_34583208batch_normalization_34583210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345826362-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_34583213dense_34583215*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_345830652
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_34583218batch_normalization_1_34583220batch_normalization_1_34583222batch_normalization_1_34583224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827982/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_34583227dense_1_34583229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_345830912!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_34583232batch_normalization_2_34583234batch_normalization_2_34583236batch_normalization_2_34583238*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829602/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_34584348
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:	?

unknown_18:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_345834742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34583968

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
I__inference_log_std_dev_layer_call_and_return_conditional_losses_34583462

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?e
?
H__inference_sequential_layer_call_and_return_conditional_losses_34584555

inputs>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:@
2batch_normalization_cast_2_readvariableop_resource:@
2batch_normalization_cast_3_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?C
4batch_normalization_1_cast_2_readvariableop_resource:	?C
4batch_normalization_1_cast_3_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?A
2batch_normalization_2_cast_readvariableop_resource:	?C
4batch_normalization_2_cast_1_readvariableop_resource:	?C
4batch_normalization_2_cast_2_readvariableop_resource:	?C
4batch_normalization_2_cast_3_readvariableop_resource:	?
identity??'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?)batch_normalization/Cast_2/ReadVariableOp?)batch_normalization/Cast_3/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?+batch_normalization_2/Cast_2/ReadVariableOp?+batch_normalization_2/Cast_3/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp?
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/add_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp?
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOp?
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOp?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/Rsqrt?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/mul_1?
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/mul_2?
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_34584663

inputsI
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:>
0batch_normalization_cast_readvariableop_resource:@
2batch_normalization_cast_1_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_2_cast_readvariableop_resource:	?C
4batch_normalization_2_cast_1_readvariableop_resource:	?
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul?
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg?
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mul?
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMeandense/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/mul?
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg?
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/mul?
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_1/batchnorm/add_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices?
"batch_normalization_2/moments/meanMeandense_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2$
"batch_normalization_2/moments/mean?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?2,
*batch_normalization_2/moments/StopGradient?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????21
/batch_normalization_2/moments/SquaredDifference?
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2(
&batch_normalization_2/moments/variance?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1?
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_2/AssignMovingAvg/decay?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_2/AssignMovingAvg/sub?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_2/AssignMovingAvg/mul?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg?
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_2/AssignMovingAvg_1/decay?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_2/AssignMovingAvg_1/sub?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_2/AssignMovingAvg_1/mul?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/Rsqrt?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/mul_1?
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_2/batchnorm/mul_2?
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_call_5602484

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_call_5600075

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
__inference_logprob_5602722
observations	
value&
continuous_actor_5602654:&
continuous_actor_5602656:&
continuous_actor_5602658:&
continuous_actor_5602660:+
continuous_actor_5602662:	?'
continuous_actor_5602664:	?'
continuous_actor_5602666:	?'
continuous_actor_5602668:	?'
continuous_actor_5602670:	?'
continuous_actor_5602672:	?,
continuous_actor_5602674:
??'
continuous_actor_5602676:	?'
continuous_actor_5602678:	?'
continuous_actor_5602680:	?'
continuous_actor_5602682:	?'
continuous_actor_5602684:	?+
continuous_actor_5602686:	?&
continuous_actor_5602688:+
continuous_actor_5602690:	?&
continuous_actor_5602692:
identity??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallobservationscontinuous_actor_5602654continuous_actor_5602656continuous_actor_5602658continuous_actor_5602660continuous_actor_5602662continuous_actor_5602664continuous_actor_5602666continuous_actor_5602668continuous_actor_5602670continuous_actor_5602672continuous_actor_5602674continuous_actor_5602676continuous_actor_5602678continuous_actor_5602680continuous_actor_5602682continuous_actor_5602684continuous_actor_5602686continuous_actor_5602688continuous_actor_5602690continuous_actor_5602692* 
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	?:	?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_56022862*
(continuous_actor/StatefulPartitionedCally
subSubvalue1continuous_actor/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:
?2
subn
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:	?2
ExpS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/yV
addAddV2Exp:y:0add/y:output:0*
T0*
_output_shapes
:	?2
add]
truedivRealDivsub:z:0add:z:0*
T0*#
_output_shapes
:
?2	
truedivS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowtruediv:z:0pow/y:output:0*
T0*#
_output_shapes
:
?2
powS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/x~
mulMulmul/x:output:01continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:	?2
mulW
add_1AddV2pow:z:0mul:z:0*
T0*#
_output_shapes
:
?2
add_1W
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *????2	
add_2/yb
add_2AddV2	add_1:z:0add_2/y:output:0*
T0*#
_output_shapes
:
?2
add_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x`
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*#
_output_shapes
:
?2
mul_1W
add_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *r1?2	
add_3/x^
add_3AddV2add_3/x:output:0value*
T0*#
_output_shapes
:
?2
add_3W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_2/x\
mul_2Mulmul_2/x:output:0value*
T0*#
_output_shapes
:
?2
mul_2Y
SoftplusSoftplus	mul_2:z:0*
T0*#
_output_shapes
:
?2

Softplush
add_4AddV2	add_3:z:0Softplus:activations:0*
T0*#
_output_shapes
:
?2
add_4W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_3/x`
mul_3Mulmul_3/x:output:0	add_4:z:0*
T0*#
_output_shapes
:
?2
mul_3Y
sub_1Sub	mul_1:z:0	mul_3:z:0*
T0*#
_output_shapes
:
?2
sub_1y
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicesf
SumSum	sub_1:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:	
?2
Sum_
IdentityIdentitySum:output:0^NoOp*
T0*
_output_shapes
:	
?2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:	?:
?: : : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:M I

_output_shapes
:	?
&
_user_specified_nameobservations:JF
#
_output_shapes
:
?

_user_specified_namevalue
?)
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34582960

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584093

inputsT
Fsequential_batch_normalization_assignmovingavg_readvariableop_resource:V
Hsequential_batch_normalization_assignmovingavg_1_readvariableop_resource:I
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_1_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_2_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?.sequential/batch_normalization/AssignMovingAvg?=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?0sequential/batch_normalization/AssignMovingAvg_1??sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?0sequential/batch_normalization_1/AssignMovingAvg??sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_1/AssignMovingAvg_1?Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?0sequential/batch_normalization_2/AssignMovingAvg??sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_2/AssignMovingAvg_1?Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
=sequential/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential/batch_normalization/moments/mean/reduction_indices?
+sequential/batch_normalization/moments/meanMeaninputsFsequential/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2-
+sequential/batch_normalization/moments/mean?
3sequential/batch_normalization/moments/StopGradientStopGradient4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:25
3sequential/batch_normalization/moments/StopGradient?
8sequential/batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs<sequential/batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2:
8sequential/batch_normalization/moments/SquaredDifference?
Asequential/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/batch_normalization/moments/variance/reduction_indices?
/sequential/batch_normalization/moments/varianceMean<sequential/batch_normalization/moments/SquaredDifference:z:0Jsequential/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(21
/sequential/batch_normalization/moments/variance?
.sequential/batch_normalization/moments/SqueezeSqueeze4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 20
.sequential/batch_normalization/moments/Squeeze?
0sequential/batch_normalization/moments/Squeeze_1Squeeze8sequential/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 22
0sequential/batch_normalization/moments/Squeeze_1?
4sequential/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential/batch_normalization/AssignMovingAvg/decay?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?
2sequential/batch_normalization/AssignMovingAvg/subSubEsequential/batch_normalization/AssignMovingAvg/ReadVariableOp:value:07sequential/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:24
2sequential/batch_normalization/AssignMovingAvg/sub?
2sequential/batch_normalization/AssignMovingAvg/mulMul6sequential/batch_normalization/AssignMovingAvg/sub:z:0=sequential/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:24
2sequential/batch_normalization/AssignMovingAvg/mul?
.sequential/batch_normalization/AssignMovingAvgAssignSubVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource6sequential/batch_normalization/AssignMovingAvg/mul:z:0>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype020
.sequential/batch_normalization/AssignMovingAvg?
6sequential/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization/AssignMovingAvg_1/decay?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?
4sequential/batch_normalization/AssignMovingAvg_1/subSubGsequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:09sequential/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:26
4sequential/batch_normalization/AssignMovingAvg_1/sub?
4sequential/batch_normalization/AssignMovingAvg_1/mulMul8sequential/batch_normalization/AssignMovingAvg_1/sub:z:0?sequential/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:26
4sequential/batch_normalization/AssignMovingAvg_1/mul?
0sequential/batch_normalization/AssignMovingAvg_1AssignSubVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource8sequential/batch_normalization/AssignMovingAvg_1/mul:z:0@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization/AssignMovingAvg_1?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV29sequential/batch_normalization/moments/Squeeze_1:output:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul7sequential/batch_normalization/moments/Squeeze:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub:sequential/batch_normalization/Cast/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
?sequential/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_1/moments/mean/reduction_indices?
-sequential/batch_normalization_1/moments/meanMean#sequential/dense/Relu:activations:0Hsequential/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_1/moments/mean?
5sequential/batch_normalization_1/moments/StopGradientStopGradient6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_1/moments/StopGradient?
:sequential/batch_normalization_1/moments/SquaredDifferenceSquaredDifference#sequential/dense/Relu:activations:0>sequential/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_1/moments/SquaredDifference?
Csequential/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_1/moments/variance/reduction_indices?
1sequential/batch_normalization_1/moments/varianceMean>sequential/batch_normalization_1/moments/SquaredDifference:z:0Lsequential/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_1/moments/variance?
0sequential/batch_normalization_1/moments/SqueezeSqueeze6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_1/moments/Squeeze?
2sequential/batch_normalization_1/moments/Squeeze_1Squeeze:sequential/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_1/moments/Squeeze_1?
6sequential/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_1/AssignMovingAvg/decay?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_1/AssignMovingAvg/subSubGsequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/sub?
4sequential/batch_normalization_1/AssignMovingAvg/mulMul8sequential/batch_normalization_1/AssignMovingAvg/sub:z:0?sequential/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/mul?
0sequential/batch_normalization_1/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource8sequential/batch_normalization_1/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_1/AssignMovingAvg?
8sequential/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_1/AssignMovingAvg_1/decay?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_1/AssignMovingAvg_1/subSubIsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/sub?
6sequential/batch_normalization_1/AssignMovingAvg_1/mulMul:sequential/batch_normalization_1/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/mul?
2sequential/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_1/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_1/AssignMovingAvg_1?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2;sequential/batch_normalization_1/moments/Squeeze_1:output:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul9sequential/batch_normalization_1/moments/Squeeze:output:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub<sequential/batch_normalization_1/Cast/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
?sequential/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_2/moments/mean/reduction_indices?
-sequential/batch_normalization_2/moments/meanMean%sequential/dense_1/Relu:activations:0Hsequential/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_2/moments/mean?
5sequential/batch_normalization_2/moments/StopGradientStopGradient6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_2/moments/StopGradient?
:sequential/batch_normalization_2/moments/SquaredDifferenceSquaredDifference%sequential/dense_1/Relu:activations:0>sequential/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_2/moments/SquaredDifference?
Csequential/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_2/moments/variance/reduction_indices?
1sequential/batch_normalization_2/moments/varianceMean>sequential/batch_normalization_2/moments/SquaredDifference:z:0Lsequential/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_2/moments/variance?
0sequential/batch_normalization_2/moments/SqueezeSqueeze6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_2/moments/Squeeze?
2sequential/batch_normalization_2/moments/Squeeze_1Squeeze:sequential/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_2/moments/Squeeze_1?
6sequential/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_2/AssignMovingAvg/decay?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_2/AssignMovingAvg/subSubGsequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/sub?
4sequential/batch_normalization_2/AssignMovingAvg/mulMul8sequential/batch_normalization_2/AssignMovingAvg/sub:z:0?sequential/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/mul?
0sequential/batch_normalization_2/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource8sequential/batch_normalization_2/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_2/AssignMovingAvg?
8sequential/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_2/AssignMovingAvg_1/decay?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_2/AssignMovingAvg_1/subSubIsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/sub?
6sequential/batch_normalization_2/AssignMovingAvg_1/mulMul:sequential/batch_normalization_2/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/mul?
2sequential/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_2/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_2/AssignMovingAvg_1?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2;sequential/batch_normalization_2/moments/Squeeze_1:output:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul9sequential/batch_normalization_2/moments/Squeeze:output:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub<sequential/batch_normalization_2/Cast/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp/^sequential/batch_normalization/AssignMovingAvg>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp1^sequential/batch_normalization/AssignMovingAvg_1@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp1^sequential/batch_normalization_1/AssignMovingAvg@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_1/AssignMovingAvg_1B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp1^sequential/batch_normalization_2/AssignMovingAvg@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_2/AssignMovingAvg_1B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2`
.sequential/batch_normalization/AssignMovingAvg.sequential/batch_normalization/AssignMovingAvg2~
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp2d
0sequential/batch_normalization/AssignMovingAvg_10sequential/batch_normalization/AssignMovingAvg_12?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_1/AssignMovingAvg0sequential/batch_normalization_1/AssignMovingAvg2?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_1/AssignMovingAvg_12sequential/batch_normalization_1/AssignMovingAvg_12?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_2/AssignMovingAvg0sequential/batch_normalization_2/AssignMovingAvg2?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_2/AssignMovingAvg_12sequential/batch_normalization_2/AssignMovingAvg_12?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_action_5602344
observations&
continuous_actor_5602287:&
continuous_actor_5602289:&
continuous_actor_5602291:&
continuous_actor_5602293:+
continuous_actor_5602295:	?'
continuous_actor_5602297:	?'
continuous_actor_5602299:	?'
continuous_actor_5602301:	?'
continuous_actor_5602303:	?'
continuous_actor_5602305:	?,
continuous_actor_5602307:
??'
continuous_actor_5602309:	?'
continuous_actor_5602311:	?'
continuous_actor_5602313:	?'
continuous_actor_5602315:	?'
continuous_actor_5602317:	?+
continuous_actor_5602319:	?&
continuous_actor_5602321:+
continuous_actor_5602323:	?&
continuous_actor_5602325:
identity??(continuous_actor/StatefulPartitionedCall?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallobservationscontinuous_actor_5602287continuous_actor_5602289continuous_actor_5602291continuous_actor_5602293continuous_actor_5602295continuous_actor_5602297continuous_actor_5602299continuous_actor_5602301continuous_actor_5602303continuous_actor_5602305continuous_actor_5602307continuous_actor_5602309continuous_actor_5602311continuous_actor_5602313continuous_actor_5602315continuous_actor_5602317continuous_actor_5602319continuous_actor_5602321continuous_actor_5602323continuous_actor_5602325* 
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	?:	?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_56022862*
(continuous_actor/StatefulPartitionedCalln
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:	?2
Exp{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal/shape?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes
:	?*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0Exp:y:0*
T0*
_output_shapes
:	?2
random_normal/mul?
random_normalAddV2random_normal/mul:z:01continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
random_normalQ
TanhTanhrandom_normal:z:0*
T0*
_output_shapes
:	?2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes
:	?2
addc
mul/xConst*
_output_shapes
:*
dtype0*!
valueB"   @   @   @2
mul/xT
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes
:	?2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yd
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes
:	?2	
truedivg
add_1/xConst*
_output_shapes
:*
dtype0*!
valueB"  ??  ??  ??2	
add_1/x`
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes
:	?2
add_1\
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
:	?2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	?: : : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:M I

_output_shapes
:	?
&
_user_specified_nameobservations
?
?
&__inference_signature_wrapper_34583885
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:	?

unknown_18:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_345825522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_mean_layer_call_and_return_conditional_losses_34584747

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs

?
__inference_call_5602143

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes

:20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes

:20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueg
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34584995

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34582900

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
H__inference_sequential_layer_call_and_return_conditional_losses_34583107

inputs*
batch_normalization_34583045:*
batch_normalization_34583047:*
batch_normalization_34583049:*
batch_normalization_34583051:!
dense_34583066:	?
dense_34583068:	?-
batch_normalization_1_34583071:	?-
batch_normalization_1_34583073:	?-
batch_normalization_1_34583075:	?-
batch_normalization_1_34583077:	?$
dense_1_34583092:
??
dense_1_34583094:	?-
batch_normalization_2_34583097:	?-
batch_normalization_2_34583099:	?-
batch_normalization_2_34583101:	?-
batch_normalization_2_34583103:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_34583045batch_normalization_34583047batch_normalization_34583049batch_normalization_34583051*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345825762-
+batch_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_34583066dense_34583068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_345830652
dense/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_34583071batch_normalization_1_34583073batch_normalization_1_34583075batch_normalization_1_34583077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827382/
-batch_normalization_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_34583092dense_1_34583094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_345830912!
dense_1/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_34583097batch_normalization_2_34583099batch_normalization_2_34583101batch_normalization_2_34583103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829002/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_34583065

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?)
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34582798

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34582636

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_action_5602401
observations&
continuous_actor_5602348:&
continuous_actor_5602350:&
continuous_actor_5602352:&
continuous_actor_5602354:+
continuous_actor_5602356:	?'
continuous_actor_5602358:	?'
continuous_actor_5602360:	?'
continuous_actor_5602362:	?'
continuous_actor_5602364:	?'
continuous_actor_5602366:	?,
continuous_actor_5602368:
??'
continuous_actor_5602370:	?'
continuous_actor_5602372:	?'
continuous_actor_5602374:	?'
continuous_actor_5602376:	?'
continuous_actor_5602378:	?+
continuous_actor_5602380:	?&
continuous_actor_5602382:+
continuous_actor_5602384:	?&
continuous_actor_5602386:
identity??(continuous_actor/StatefulPartitionedCall|
continuous_actor/CastCastobservations*

DstT0*

SrcT0*
_output_shapes

:2
continuous_actor/Cast?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallcontinuous_actor/Cast:y:0continuous_actor_5602348continuous_actor_5602350continuous_actor_5602352continuous_actor_5602354continuous_actor_5602356continuous_actor_5602358continuous_actor_5602360continuous_actor_5602362continuous_actor_5602364continuous_actor_5602366continuous_actor_5602368continuous_actor_5602370continuous_actor_5602372continuous_actor_5602374continuous_actor_5602376continuous_actor_5602378continuous_actor_5602380continuous_actor_5602382continuous_actor_5602384continuous_actor_5602386* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_56021432*
(continuous_actor/StatefulPartitionedCallm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
Expp
TanhTanh1continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yV
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes

:2
addc
mul/xConst*
_output_shapes
:*
dtype0*!
valueB"   @   @   @2
mul/xS
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes

:2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yc
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivg
add_1/xConst*
_output_shapes
:*
dtype0*!
valueB"  ??  ??  ??2	
add_1/x_
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
add_1[
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes

:2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:L H

_output_shapes

:
&
_user_specified_nameobservations
?
?
-__inference_sequential_layer_call_fn_34584700

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345831072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584176
input_1I
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinput_10sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_34583091

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_34583314
batch_normalization_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345832422
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_namebatch_normalization_input
?
?
-__inference_sequential_layer_call_fn_34584737

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345832422
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?)
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34585029

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
I__inference_log_std_dev_layer_call_and_return_conditional_losses_34584766

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_34584966

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_34584442

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:	?

unknown_18:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_345836402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_34584975

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_345830912
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
؎
?
__inference_call_5602286

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	?20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0* 
_output_shapes
:
??22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	?2
clip_by_valueh
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes
:	?2

Identityh

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes
:	?2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:	?: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_34584855

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_345826362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_34584489
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:	?

unknown_18:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_345836402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?)
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584929

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_34584875

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_345830652
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_continuous_actor_layer_call_fn_34584395

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:

unknown_17:	?

unknown_18:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_continuous_actor_layer_call_and_return_conditional_losses_345834742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_34583142
batch_normalization_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345831072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_namebatch_normalization_input
?
?
8__inference_batch_normalization_1_layer_call_fn_34584942

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345827382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_2_layer_call_fn_34585055

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345829602
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_action_5602201
observations&
continuous_actor_5602144:&
continuous_actor_5602146:&
continuous_actor_5602148:&
continuous_actor_5602150:+
continuous_actor_5602152:	?'
continuous_actor_5602154:	?'
continuous_actor_5602156:	?'
continuous_actor_5602158:	?'
continuous_actor_5602160:	?'
continuous_actor_5602162:	?,
continuous_actor_5602164:
??'
continuous_actor_5602166:	?'
continuous_actor_5602168:	?'
continuous_actor_5602170:	?'
continuous_actor_5602172:	?'
continuous_actor_5602174:	?+
continuous_actor_5602176:	?&
continuous_actor_5602178:+
continuous_actor_5602180:	?&
continuous_actor_5602182:
identity??(continuous_actor/StatefulPartitionedCall|
continuous_actor/CastCastobservations*

DstT0*

SrcT0*
_output_shapes

:2
continuous_actor/Cast?
(continuous_actor/StatefulPartitionedCallStatefulPartitionedCallcontinuous_actor/Cast:y:0continuous_actor_5602144continuous_actor_5602146continuous_actor_5602148continuous_actor_5602150continuous_actor_5602152continuous_actor_5602154continuous_actor_5602156continuous_actor_5602158continuous_actor_5602160continuous_actor_5602162continuous_actor_5602164continuous_actor_5602166continuous_actor_5602168continuous_actor_5602170continuous_actor_5602172continuous_actor_5602174continuous_actor_5602176continuous_actor_5602178continuous_actor_5602180continuous_actor_5602182* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_56021432*
(continuous_actor/StatefulPartitionedCallm
ExpExp1continuous_actor/StatefulPartitionedCall:output:1*
T0*
_output_shapes

:2
Exp{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
random_normal/shape?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:*
dtype0*

seed2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0Exp:y:0*
T0*
_output_shapes

:2
random_normal/mul?
random_normalAddV2random_normal/mul:z:01continuous_actor/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
random_normalP
TanhTanhrandom_normal:z:0*
T0*
_output_shapes

:2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yV
addAddV2Tanh:y:0add/y:output:0*
T0*
_output_shapes

:2
addc
mul/xConst*
_output_shapes
:*
dtype0*!
valueB"   @   @   @2
mul/xS
mulMulmul/x:output:0add:z:0*
T0*
_output_shapes

:2
mul[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yc
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivg
add_1/xConst*
_output_shapes
:*
dtype0*!
valueB"  ??  ??  ??2	
add_1/x_
add_1AddV2add_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
add_1[
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes

:2

Identityy
NoOpNoOp)^continuous_actor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : : : 2T
(continuous_actor/StatefulPartitionedCall(continuous_actor/StatefulPartitionedCall:L H

_output_shapes

:
&
_user_specified_nameobservations
?

?
B__inference_mean_layer_call_and_return_conditional_losses_34583446

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34583640

inputs!
sequential_34583591:!
sequential_34583593:!
sequential_34583595:!
sequential_34583597:&
sequential_34583599:	?"
sequential_34583601:	?"
sequential_34583603:	?"
sequential_34583605:	?"
sequential_34583607:	?"
sequential_34583609:	?'
sequential_34583611:
??"
sequential_34583613:	?"
sequential_34583615:	?"
sequential_34583617:	?"
sequential_34583619:	?"
sequential_34583621:	? 
mean_34583624:	?
mean_34583626:'
log_std_dev_34583629:	?"
log_std_dev_34583631:
identity

identity_1??#log_std_dev/StatefulPartitionedCall?mean/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_34583591sequential_34583593sequential_34583595sequential_34583597sequential_34583599sequential_34583601sequential_34583603sequential_34583605sequential_34583607sequential_34583609sequential_34583611sequential_34583613sequential_34583615sequential_34583617sequential_34583619sequential_34583621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_345832422$
"sequential/StatefulPartitionedCall?
mean/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0mean_34583624mean_34583626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_mean_layer_call_and_return_conditional_losses_345834462
mean/StatefulPartitionedCall?
#log_std_dev/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0log_std_dev_34583629log_std_dev_34583631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_345834622%
#log_std_dev/StatefulPartitionedCallw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum,log_std_dev/StatefulPartitionedCall:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value?
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp$^log_std_dev/StatefulPartitionedCall^mean/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2J
#log_std_dev/StatefulPartitionedCall#log_std_dev/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_log_std_dev_layer_call_fn_34584775

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_log_std_dev_layer_call_and_return_conditional_losses_345834622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584301
input_1T
Fsequential_batch_normalization_assignmovingavg_readvariableop_resource:V
Hsequential_batch_normalization_assignmovingavg_1_readvariableop_resource:I
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_1_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?W
Hsequential_batch_normalization_2_assignmovingavg_readvariableop_resource:	?Y
Jsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?.sequential/batch_normalization/AssignMovingAvg?=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?0sequential/batch_normalization/AssignMovingAvg_1??sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?0sequential/batch_normalization_1/AssignMovingAvg??sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_1/AssignMovingAvg_1?Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?0sequential/batch_normalization_2/AssignMovingAvg??sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?2sequential/batch_normalization_2/AssignMovingAvg_1?Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
=sequential/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential/batch_normalization/moments/mean/reduction_indices?
+sequential/batch_normalization/moments/meanMeaninput_1Fsequential/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2-
+sequential/batch_normalization/moments/mean?
3sequential/batch_normalization/moments/StopGradientStopGradient4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:25
3sequential/batch_normalization/moments/StopGradient?
8sequential/batch_normalization/moments/SquaredDifferenceSquaredDifferenceinput_1<sequential/batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2:
8sequential/batch_normalization/moments/SquaredDifference?
Asequential/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/batch_normalization/moments/variance/reduction_indices?
/sequential/batch_normalization/moments/varianceMean<sequential/batch_normalization/moments/SquaredDifference:z:0Jsequential/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(21
/sequential/batch_normalization/moments/variance?
.sequential/batch_normalization/moments/SqueezeSqueeze4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 20
.sequential/batch_normalization/moments/Squeeze?
0sequential/batch_normalization/moments/Squeeze_1Squeeze8sequential/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 22
0sequential/batch_normalization/moments/Squeeze_1?
4sequential/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential/batch_normalization/AssignMovingAvg/decay?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp?
2sequential/batch_normalization/AssignMovingAvg/subSubEsequential/batch_normalization/AssignMovingAvg/ReadVariableOp:value:07sequential/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:24
2sequential/batch_normalization/AssignMovingAvg/sub?
2sequential/batch_normalization/AssignMovingAvg/mulMul6sequential/batch_normalization/AssignMovingAvg/sub:z:0=sequential/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:24
2sequential/batch_normalization/AssignMovingAvg/mul?
.sequential/batch_normalization/AssignMovingAvgAssignSubVariableOpFsequential_batch_normalization_assignmovingavg_readvariableop_resource6sequential/batch_normalization/AssignMovingAvg/mul:z:0>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype020
.sequential/batch_normalization/AssignMovingAvg?
6sequential/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization/AssignMovingAvg_1/decay?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?
4sequential/batch_normalization/AssignMovingAvg_1/subSubGsequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:09sequential/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:26
4sequential/batch_normalization/AssignMovingAvg_1/sub?
4sequential/batch_normalization/AssignMovingAvg_1/mulMul8sequential/batch_normalization/AssignMovingAvg_1/sub:z:0?sequential/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:26
4sequential/batch_normalization/AssignMovingAvg_1/mul?
0sequential/batch_normalization/AssignMovingAvg_1AssignSubVariableOpHsequential_batch_normalization_assignmovingavg_1_readvariableop_resource8sequential/batch_normalization/AssignMovingAvg_1/mul:z:0@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization/AssignMovingAvg_1?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV29sequential/batch_normalization/moments/Squeeze_1:output:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinput_10sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul7sequential/batch_normalization/moments/Squeeze:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub:sequential/batch_normalization/Cast/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
?sequential/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_1/moments/mean/reduction_indices?
-sequential/batch_normalization_1/moments/meanMean#sequential/dense/Relu:activations:0Hsequential/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_1/moments/mean?
5sequential/batch_normalization_1/moments/StopGradientStopGradient6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_1/moments/StopGradient?
:sequential/batch_normalization_1/moments/SquaredDifferenceSquaredDifference#sequential/dense/Relu:activations:0>sequential/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_1/moments/SquaredDifference?
Csequential/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_1/moments/variance/reduction_indices?
1sequential/batch_normalization_1/moments/varianceMean>sequential/batch_normalization_1/moments/SquaredDifference:z:0Lsequential/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_1/moments/variance?
0sequential/batch_normalization_1/moments/SqueezeSqueeze6sequential/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_1/moments/Squeeze?
2sequential/batch_normalization_1/moments/Squeeze_1Squeeze:sequential/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_1/moments/Squeeze_1?
6sequential/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_1/AssignMovingAvg/decay?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_1/AssignMovingAvg/subSubGsequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/sub?
4sequential/batch_normalization_1/AssignMovingAvg/mulMul8sequential/batch_normalization_1/AssignMovingAvg/sub:z:0?sequential/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_1/AssignMovingAvg/mul?
0sequential/batch_normalization_1/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_1_assignmovingavg_readvariableop_resource8sequential/batch_normalization_1/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_1/AssignMovingAvg?
8sequential/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_1/AssignMovingAvg_1/decay?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_1/AssignMovingAvg_1/subSubIsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/sub?
6sequential/batch_normalization_1/AssignMovingAvg_1/mulMul:sequential/batch_normalization_1/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_1/AssignMovingAvg_1/mul?
2sequential/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_1_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_1/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_1/AssignMovingAvg_1?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2;sequential/batch_normalization_1/moments/Squeeze_1:output:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul9sequential/batch_normalization_1/moments/Squeeze:output:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub<sequential/batch_normalization_1/Cast/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
?sequential/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/batch_normalization_2/moments/mean/reduction_indices?
-sequential/batch_normalization_2/moments/meanMean%sequential/dense_1/Relu:activations:0Hsequential/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2/
-sequential/batch_normalization_2/moments/mean?
5sequential/batch_normalization_2/moments/StopGradientStopGradient6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	?27
5sequential/batch_normalization_2/moments/StopGradient?
:sequential/batch_normalization_2/moments/SquaredDifferenceSquaredDifference%sequential/dense_1/Relu:activations:0>sequential/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2<
:sequential/batch_normalization_2/moments/SquaredDifference?
Csequential/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/batch_normalization_2/moments/variance/reduction_indices?
1sequential/batch_normalization_2/moments/varianceMean>sequential/batch_normalization_2/moments/SquaredDifference:z:0Lsequential/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(23
1sequential/batch_normalization_2/moments/variance?
0sequential/batch_normalization_2/moments/SqueezeSqueeze6sequential/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 22
0sequential/batch_normalization_2/moments/Squeeze?
2sequential/batch_normalization_2/moments/Squeeze_1Squeeze:sequential/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 24
2sequential/batch_normalization_2/moments/Squeeze_1?
6sequential/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential/batch_normalization_2/AssignMovingAvg/decay?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?
4sequential/batch_normalization_2/AssignMovingAvg/subSubGsequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:09sequential/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/sub?
4sequential/batch_normalization_2/AssignMovingAvg/mulMul8sequential/batch_normalization_2/AssignMovingAvg/sub:z:0?sequential/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?26
4sequential/batch_normalization_2/AssignMovingAvg/mul?
0sequential/batch_normalization_2/AssignMovingAvgAssignSubVariableOpHsequential_batch_normalization_2_assignmovingavg_readvariableop_resource8sequential/batch_normalization_2/AssignMovingAvg/mul:z:0@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype022
0sequential/batch_normalization_2/AssignMovingAvg?
8sequential/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential/batch_normalization_2/AssignMovingAvg_1/decay?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
6sequential/batch_normalization_2/AssignMovingAvg_1/subSubIsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0;sequential/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/sub?
6sequential/batch_normalization_2/AssignMovingAvg_1/mulMul:sequential/batch_normalization_2/AssignMovingAvg_1/sub:z:0Asequential/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?28
6sequential/batch_normalization_2/AssignMovingAvg_1/mul?
2sequential/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpJsequential_batch_normalization_2_assignmovingavg_1_readvariableop_resource:sequential/batch_normalization_2/AssignMovingAvg_1/mul:z:0B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype024
2sequential/batch_normalization_2/AssignMovingAvg_1?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2;sequential/batch_normalization_2/moments/Squeeze_1:output:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul9sequential/batch_normalization_2/moments/Squeeze:output:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub<sequential/batch_normalization_2/Cast/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuep
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityp

Identity_1Identityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp/^sequential/batch_normalization/AssignMovingAvg>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp1^sequential/batch_normalization/AssignMovingAvg_1@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp1^sequential/batch_normalization_1/AssignMovingAvg@^sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_1/AssignMovingAvg_1B^sequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp1^sequential/batch_normalization_2/AssignMovingAvg@^sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp3^sequential/batch_normalization_2/AssignMovingAvg_1B^sequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2`
.sequential/batch_normalization/AssignMovingAvg.sequential/batch_normalization/AssignMovingAvg2~
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp2d
0sequential/batch_normalization/AssignMovingAvg_10sequential/batch_normalization/AssignMovingAvg_12?
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_1/AssignMovingAvg0sequential/batch_normalization_1/AssignMovingAvg2?
?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_1/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_1/AssignMovingAvg_12sequential/batch_normalization_1/AssignMovingAvg_12?
Asequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2d
0sequential/batch_normalization_2/AssignMovingAvg0sequential/batch_normalization_2/AssignMovingAvg2?
?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp?sequential/batch_normalization_2/AssignMovingAvg/ReadVariableOp2h
2sequential/batch_normalization_2/AssignMovingAvg_12sequential/batch_normalization_2/AssignMovingAvg_12?
Asequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpAsequential/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1

?
__inference_call_5602567

inputsI
;sequential_batch_normalization_cast_readvariableop_resource:K
=sequential_batch_normalization_cast_1_readvariableop_resource:K
=sequential_batch_normalization_cast_2_readvariableop_resource:K
=sequential_batch_normalization_cast_3_readvariableop_resource:B
/sequential_dense_matmul_readvariableop_resource:	??
0sequential_dense_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_1_cast_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_1_cast_3_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?L
=sequential_batch_normalization_2_cast_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_1_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_2_readvariableop_resource:	?N
?sequential_batch_normalization_2_cast_3_readvariableop_resource:	?6
#mean_matmul_readvariableop_resource:	?2
$mean_biasadd_readvariableop_resource:=
*log_std_dev_matmul_readvariableop_resource:	?9
+log_std_dev_biasadd_readvariableop_resource:
identity

identity_1??"log_std_dev/BiasAdd/ReadVariableOp?!log_std_dev/MatMul/ReadVariableOp?mean/BiasAdd/ReadVariableOp?mean/MatMul/ReadVariableOp?2sequential/batch_normalization/Cast/ReadVariableOp?4sequential/batch_normalization/Cast_1/ReadVariableOp?4sequential/batch_normalization/Cast_2/ReadVariableOp?4sequential/batch_normalization/Cast_3/ReadVariableOp?4sequential/batch_normalization_1/Cast/ReadVariableOp?6sequential/batch_normalization_1/Cast_1/ReadVariableOp?6sequential/batch_normalization_1/Cast_2/ReadVariableOp?6sequential/batch_normalization_1/Cast_3/ReadVariableOp?4sequential/batch_normalization_2/Cast/ReadVariableOp?6sequential/batch_normalization_2/Cast_1/ReadVariableOp?6sequential/batch_normalization_2/Cast_2/ReadVariableOp?6sequential/batch_normalization_2/Cast_3/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/batch_normalization/Cast/ReadVariableOp?
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_1/ReadVariableOp?
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_2/ReadVariableOp?
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/batch_normalization/Cast_3/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/Rsqrt?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulinputs0sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes

:20
.sequential/batch_normalization/batchnorm/mul_1?
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization/batchnorm/mul_2?
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes

:20
.sequential/batch_normalization/batchnorm/add_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense/Relu?
4sequential/batch_normalization_1/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_1/Cast/ReadVariableOp?
6sequential/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_1/ReadVariableOp?
6sequential/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_2/ReadVariableOp?
6sequential/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_1/Cast_3/ReadVariableOp?
0sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_1/batchnorm/add/y?
.sequential/batch_normalization_1/batchnorm/addAddV2>sequential/batch_normalization_1/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/add?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt2sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/Rsqrt?
.sequential/batch_normalization_1/batchnorm/mulMul4sequential/batch_normalization_1/batchnorm/Rsqrt:y:0>sequential/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/mul?
0sequential/batch_normalization_1/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/mul_1?
0sequential/batch_normalization_1/batchnorm/mul_2Mul<sequential/batch_normalization_1/Cast/ReadVariableOp:value:02sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_1/batchnorm/mul_2?
.sequential/batch_normalization_1/batchnorm/subSub>sequential/batch_normalization_1/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_1/batchnorm/sub?
0sequential/batch_normalization_1/batchnorm/add_1AddV24sequential/batch_normalization_1/batchnorm/mul_1:z:02sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_1/batchnorm/add_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul4sequential/batch_normalization_1/batchnorm/add_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/dense_1/Relu?
4sequential/batch_normalization_2/Cast/ReadVariableOpReadVariableOp=sequential_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:?*
dtype026
4sequential/batch_normalization_2/Cast/ReadVariableOp?
6sequential/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_1/ReadVariableOp?
6sequential/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_2/ReadVariableOp?
6sequential/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp?sequential_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/batch_normalization_2/Cast_3/ReadVariableOp?
0sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0sequential/batch_normalization_2/batchnorm/add/y?
.sequential/batch_normalization_2/batchnorm/addAddV2>sequential/batch_normalization_2/Cast_1/ReadVariableOp:value:09sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/add?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt2sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/Rsqrt?
.sequential/batch_normalization_2/batchnorm/mulMul4sequential/batch_normalization_2/batchnorm/Rsqrt:y:0>sequential/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/mul?
0sequential/batch_normalization_2/batchnorm/mul_1Mul%sequential/dense_1/Relu:activations:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/mul_1?
0sequential/batch_normalization_2/batchnorm/mul_2Mul<sequential/batch_normalization_2/Cast/ReadVariableOp:value:02sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:?22
0sequential/batch_normalization_2/batchnorm/mul_2?
.sequential/batch_normalization_2/batchnorm/subSub>sequential/batch_normalization_2/Cast_2/ReadVariableOp:value:04sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization_2/batchnorm/sub?
0sequential/batch_normalization_2/batchnorm/add_1AddV24sequential/batch_normalization_2/batchnorm/mul_1:z:02sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*
_output_shapes
:	?22
0sequential/batch_normalization_2/batchnorm/add_1?
mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mean/MatMul/ReadVariableOp?
mean/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0"mean/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/MatMul?
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mean/BiasAdd/ReadVariableOp?
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
mean/BiasAdd?
!log_std_dev/MatMul/ReadVariableOpReadVariableOp*log_std_dev_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!log_std_dev/MatMul/ReadVariableOp?
log_std_dev/MatMulMatMul4sequential/batch_normalization_2/batchnorm/add_1:z:0)log_std_dev/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
log_std_dev/MatMul?
"log_std_dev/BiasAdd/ReadVariableOpReadVariableOp+log_std_dev_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"log_std_dev/BiasAdd/ReadVariableOp?
log_std_dev/BiasAddBiasAddlog_std_dev/MatMul:product:0*log_std_dev/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
log_std_dev/BiasAddw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumlog_std_dev/BiasAdd:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:2
clip_by_valueg
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identityg

Identity_1Identityclip_by_value:z:0^NoOp*
T0*
_output_shapes

:2

Identity_1?
NoOpNoOp#^log_std_dev/BiasAdd/ReadVariableOp"^log_std_dev/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp5^sequential/batch_normalization_1/Cast/ReadVariableOp7^sequential/batch_normalization_1/Cast_1/ReadVariableOp7^sequential/batch_normalization_1/Cast_2/ReadVariableOp7^sequential/batch_normalization_1/Cast_3/ReadVariableOp5^sequential/batch_normalization_2/Cast/ReadVariableOp7^sequential/batch_normalization_2/Cast_1/ReadVariableOp7^sequential/batch_normalization_2/Cast_2/ReadVariableOp7^sequential/batch_normalization_2/Cast_3/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : : : 2H
"log_std_dev/BiasAdd/ReadVariableOp"log_std_dev/BiasAdd/ReadVariableOp2F
!log_std_dev/MatMul/ReadVariableOp!log_std_dev/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_1/Cast/ReadVariableOp4sequential/batch_normalization_1/Cast/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_1/ReadVariableOp6sequential/batch_normalization_1/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_2/ReadVariableOp6sequential/batch_normalization_1/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_1/Cast_3/ReadVariableOp6sequential/batch_normalization_1/Cast_3/ReadVariableOp2l
4sequential/batch_normalization_2/Cast/ReadVariableOp4sequential/batch_normalization_2/Cast/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_1/ReadVariableOp6sequential/batch_normalization_2/Cast_1/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_2/ReadVariableOp6sequential/batch_normalization_2/Cast_2/ReadVariableOp2p
6sequential/batch_normalization_2/Cast_3/ReadVariableOp6sequential/batch_normalization_2/Cast_3/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584895

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:ǻ
?
	_body
_mu
_log_std
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*r&call_and_return_all_conditional_losses
s__call__
t_default_save_signature

uaction
vcall
wlogprob
xsample_logprob"
_tf_keras_model
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"
_tf_keras_sequential
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"
_tf_keras_layer
 "
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
16
17
18
19"
trackable_list_wrapper
?
0
1
"2
#3
$4
%5
(6
)7
*8
+9
10
11
12
13"
trackable_list_wrapper
?

.layers
regularization_losses
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
1metrics
2non_trainable_variables
s__call__
t_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
?
3axis
	gamma
beta
 moving_mean
!moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

"kernel
#bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
<axis
	$gamma
%beta
&moving_mean
'moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

(kernel
)bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Eaxis
	*gamma
+beta
,moving_mean
-moving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15"
trackable_list_wrapper
f
0
1
"2
#3
$4
%5
(6
)7
*8
+9"
trackable_list_wrapper
?

Jlayers
regularization_losses
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
Mmetrics
Nnon_trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:	?2mean/kernel
:2	mean/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Olayers
regularization_losses
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
Rmetrics
Snon_trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
%:#	?2log_std_dev/kernel
:2log_std_dev/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Tlayers
regularization_losses
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
Wmetrics
Xnon_trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
:	?2dense/kernel
:?2
dense/bias
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
": 
??2dense_1/kernel
:?2dense_1/bias
*:(?2batch_normalization_2/gamma
):'?2batch_normalization_2/beta
2:0? (2!batch_normalization_2/moving_mean
6:4? (2%batch_normalization_2/moving_variance
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
 0
!1
&2
'3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ylayers
4regularization_losses
Zlayer_regularization_losses
[layer_metrics
5	variables
6trainable_variables
\metrics
]non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?

^layers
8regularization_losses
_layer_regularization_losses
`layer_metrics
9	variables
:trainable_variables
ametrics
bnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?

clayers
=regularization_losses
dlayer_regularization_losses
elayer_metrics
>	variables
?trainable_variables
fmetrics
gnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

hlayers
Aregularization_losses
ilayer_regularization_losses
jlayer_metrics
B	variables
Ctrainable_variables
kmetrics
lnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?

mlayers
Fregularization_losses
nlayer_regularization_losses
olayer_metrics
G	variables
Htrainable_variables
pmetrics
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
	0

1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
 0
!1
&2
'3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?2?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34583968
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584093
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584176
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584301?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_continuous_actor_layer_call_fn_34584348
3__inference_continuous_actor_layer_call_fn_34584395
3__inference_continuous_actor_layer_call_fn_34584442
3__inference_continuous_actor_layer_call_fn_34584489?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__wrapped_model_34582552input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_action_5602201
__inference_action_5602344
__inference_action_5602401?
???
FullArgSpec4
args,?)
jself
jobservations
jdeterministic
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_5602484
__inference_call_5602567
__inference_call_5602650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_logprob_5602722?
???
FullArgSpec,
args$?!
jself
jobservations
jvalue
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec0
args(?%
jself
jobservations
j	n_samples
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_34584555
H__inference_sequential_layer_call_and_return_conditional_losses_34584663
H__inference_sequential_layer_call_and_return_conditional_losses_34583355
H__inference_sequential_layer_call_and_return_conditional_losses_34583396?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_layer_call_fn_34583142
-__inference_sequential_layer_call_fn_34584700
-__inference_sequential_layer_call_fn_34584737
-__inference_sequential_layer_call_fn_34583314?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_mean_layer_call_and_return_conditional_losses_34584747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_mean_layer_call_fn_34584756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_log_std_dev_layer_call_and_return_conditional_losses_34584766?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_log_std_dev_layer_call_fn_34584775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_34583885input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584795
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584829?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_layer_call_fn_34584842
6__inference_batch_normalization_layer_call_fn_34584855?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_34584866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_34584875?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584895
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584929?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_1_layer_call_fn_34584942
8__inference_batch_normalization_1_layer_call_fn_34584955?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_34584966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_34584975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34584995
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34585029?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_2_layer_call_fn_34585042
8__inference_batch_normalization_2_layer_call_fn_34585055?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
#__inference__wrapped_model_34582552? !"#&'%$(),-+*0?-
&?#
!?
input_1?????????
? "c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????w
__inference_action_5602201Y !"#&'%$(),-+*0?-
&?#
?
observations
p 
? "?y
__inference_action_5602344[ !"#&'%$(),-+*1?.
'?$
?
observations	?
p 
? "?	?w
__inference_action_5602401Y !"#&'%$(),-+*0?-
&?#
?
observations
p
? "??
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584895d&'%$4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34584929d&'%$4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_1_layer_call_fn_34584942W&'%$4?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_1_layer_call_fn_34584955W&'%$4?1
*?'
!?
inputs??????????
p
? "????????????
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34584995d,-+*4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34585029d,-+*4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_2_layer_call_fn_34585042W,-+*4?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_2_layer_call_fn_34585055W,-+*4?1
*?'
!?
inputs??????????
p
? "????????????
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584795b !3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34584829b !3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
6__inference_batch_normalization_layer_call_fn_34584842U !3?0
)?&
 ?
inputs?????????
p 
? "???????????
6__inference_batch_normalization_layer_call_fn_34584855U !3?0
)?&
 ?
inputs?????????
p
? "???????????
__inference_call_5602484? !"#&'%$(),-+*/?,
%?"
 ?
inputs?????????
? "=?:
?
0?????????
?
1??????????
__inference_call_5602567k !"#&'%$(),-+*&?#
?
?
inputs
? "+?(
?
0
?
1?
__inference_call_5602650n !"#&'%$(),-+*'?$
?
?
inputs	?
? "-?*
?
0	?
?
1	??
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34583968? !"#&'%$(),-+*3?0
)?&
 ?
inputs?????????
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584093? !"#&'%$(),-+*3?0
)?&
 ?
inputs?????????
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584176? !"#&'%$(),-+*4?1
*?'
!?
input_1?????????
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
N__inference_continuous_actor_layer_call_and_return_conditional_losses_34584301? !"#&'%$(),-+*4?1
*?'
!?
input_1?????????
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
3__inference_continuous_actor_layer_call_fn_34584348? !"#&'%$(),-+*4?1
*?'
!?
input_1?????????
p 
? "=?:
?
0?????????
?
1??????????
3__inference_continuous_actor_layer_call_fn_34584395? !"#&'%$(),-+*3?0
)?&
 ?
inputs?????????
p 
? "=?:
?
0?????????
?
1??????????
3__inference_continuous_actor_layer_call_fn_34584442? !"#&'%$(),-+*3?0
)?&
 ?
inputs?????????
p
? "=?:
?
0?????????
?
1??????????
3__inference_continuous_actor_layer_call_fn_34584489? !"#&'%$(),-+*4?1
*?'
!?
input_1?????????
p
? "=?:
?
0?????????
?
1??????????
E__inference_dense_1_layer_call_and_return_conditional_losses_34584966^()0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_1_layer_call_fn_34584975Q()0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_34584866]"#/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? |
(__inference_dense_layer_call_fn_34584875P"#/?,
%?"
 ?
inputs?????????
? "????????????
I__inference_log_std_dev_layer_call_and_return_conditional_losses_34584766]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
.__inference_log_std_dev_layer_call_fn_34584775P0?-
&?#
!?
inputs??????????
? "???????????
__inference_logprob_5602722t !"#&'%$(),-+*J?G
@?=
?
observations	?
?
value
?
? "?	
??
B__inference_mean_layer_call_and_return_conditional_losses_34584747]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_mean_layer_call_fn_34584756P0?-
&?#
!?
inputs??????????
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_34583355? !"#&'%$(),-+*J?G
@?=
3?0
batch_normalization_input?????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_34583396? !"#&'%$(),-+*J?G
@?=
3?0
batch_normalization_input?????????
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_34584555s !"#&'%$(),-+*7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_34584663s !"#&'%$(),-+*7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
-__inference_sequential_layer_call_fn_34583142y !"#&'%$(),-+*J?G
@?=
3?0
batch_normalization_input?????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_34583314y !"#&'%$(),-+*J?G
@?=
3?0
batch_normalization_input?????????
p

 
? "????????????
-__inference_sequential_layer_call_fn_34584700f !"#&'%$(),-+*7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
-__inference_sequential_layer_call_fn_34584737f !"#&'%$(),-+*7?4
-?*
 ?
inputs?????????
p

 
? "????????????
&__inference_signature_wrapper_34583885? !"#&'%$(),-+*;?8
? 
1?.
,
input_1!?
input_1?????????"c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????