import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import {
  DockerImageFunction,
  DockerImageCode,
  FunctionUrlAuthType,
} from "aws-cdk-lib/aws-lambda";
import { Bucket } from 'aws-cdk-lib/aws-s3';

export class AwsResourcesStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create an S3 bucket to be used by the Lambda functions
    const bucket = new Bucket(this, 'ml-model-bucket-22092024 ', {
      removalPolicy: cdk.RemovalPolicy.DESTROY, // Change to RETAIN for production
    });

    // Function to handle the API requests. Uses same base image, but different handler.
    const apiImageCode = DockerImageCode.fromImageAsset("../image", {
      cmd: ["app.handler"],
    });

    const apiFunction = new DockerImageFunction(this, "MLApiHandler", {
      code: apiImageCode,
      memorySize: 256,
      timeout: cdk.Duration.seconds(30),
      environment: {
        S3_BUCKET: bucket.bucketName,
      },
    });

    // Public URL for the API function.
    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE,
    });

    // Grant the API function permissions to access the S3 bucket
    bucket.grantReadWrite(apiFunction);


    // Output the URL for the API function.
    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url,
    });
  }
}
