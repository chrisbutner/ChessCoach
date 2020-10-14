pushd "%~dp0"
cpp\protobuf-3.13.0\protoc\bin\protoc.exe -I=cpp\protobuf --cpp_out=cpp\protobuf ChessCoach.proto