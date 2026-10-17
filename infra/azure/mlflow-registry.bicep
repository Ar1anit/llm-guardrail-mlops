// Azure Machine Learning workspace used only as the MLflow tracking server and model registry.
// It creates no compute. Deploy into its own resource group:
//   az group create -n rg-llm-guardrail-ml -l westeurope
//   az deployment group create -g rg-llm-guardrail-ml -f infra/azure/mlflow-registry.bicep
// The deployment outputs the MLflow tracking URI for MLFLOW_TRACKING_URI.

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Short prefix for resource names.')
@maxLength(10)
param prefix string = 'guardrail'

var suffix = uniqueString(resourceGroup().id)
var tags = {
  project: 'llm-guardrail'
  purpose: 'mlflow-registry'
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take('${prefix}ml${suffix}', 24)
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: take('${prefix}-kv-${suffix}', 24)
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-ai-${suffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: '${prefix}-mlw-${suffix}'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: 'LLM guardrail model registry'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: insights.id
    publicNetworkAccess: 'Enabled'
  }
}

output workspaceName string = workspace.name
output mlflowTrackingUri string = workspace.properties.mlFlowTrackingUri
