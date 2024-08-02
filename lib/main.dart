// ignore_for_file: library_private_types_in_public_api

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Flutter Example',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const ModelPage(),
    );
  }
}

class ModelPage extends StatefulWidget {
  const ModelPage({super.key});

  @override
  _ModelPageState createState() => _ModelPageState();
}

class _ModelPageState extends State<ModelPage> {
  OrtSession? session;
  String outputResult = "Loading...";

 List<List<double>> inputData = [
  [
    0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219, 0.219
  ]
];


  @override
  void initState() {
    super.initState();
    _initializeOnnx();
  }

  Future<void> _initializeOnnx() async {
    try {
      OrtEnv.instance.init();

      const assetFileName = 'lib/assets/model.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();

      final sessionOptions = OrtSessionOptions();
      session = OrtSession.fromBuffer(bytes, sessionOptions);

      if (session != null) {
        final shape = [1, 13];
        final inputOrt = OrtValueTensor.createTensorWithDataList(inputData, shape);

        final inputs = {'input': inputOrt};
        final runOptions = OrtRunOptions();
        final outputs = await session!.runAsync(runOptions, inputs);


        final result = outputs?.first?.value as List<double>?; 

        inputOrt.release();
        runOptions.release();
        outputs?.forEach((element) {
          element?.release();
        });

        setState(() {
          outputResult = result?.toString() ?? "No output";
        });

        print('Hasil inferensi: $result');
      }
    } catch (e) {
      print('Error saat loading model: $e');
      setState(() {
        outputResult = 'Error saat loading model';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ONNX Model Example'),
      ),
      body: Center(
        child: Text(
          outputResult,
          style: const TextStyle(fontSize: 24),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }

  @override
  void dispose() {
    OrtEnv.instance.release();
    super.dispose();
  }
}
