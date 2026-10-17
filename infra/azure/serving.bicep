// Shadow-mode serving on Azure Container Apps, pulling from a private Azure Container Registry.
//
// First deployment (the registry must exist and hold the image before the app can start):
//   az group create -n rg-llm-guardrail-serving -l westeurope
//   az deployment group create -g rg-llm-guardrail-serving -f infra/azure/serving.bicep \
//     -p deployApp=false
//   copy the image into the registry by digest (see docs/architecture.md, "Deploying")
//   az deployment group create -g rg-llm-guardrail-serving -f infra/azure/serving.bicep \
//     -p containerImage=<registry>/guardrail-classifier@sha256:<digest> -p apiToken=<token>
// Later images are rolled out by .github/workflows/deploy.yml, not by this template. Re-running
// the template sets the image and the API token to the values passed, so pass the current ones.

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Short prefix for resource names.')
@maxLength(10)
param prefix string = 'guardrail'

@description('Create the Container App. False creates only the registry, identity, and environment.')
param deployApp bool = true

@description('Image by digest in this template\'s registry, e.g. <login server>/guardrail-classifier@sha256:...')
param containerImage string = ''

@secure()
@description('Bearer token callers present; at least 32 characters. Stored as a Container App secret.')
param apiToken string = ''

@description('Source IP ranges (CIDR) allowed to call the app. Empty allows any address.')
param allowedSourceRanges array = []

@description('Replicas kept warm. 0 scales to zero when idle; the first request then waits for a cold start.')
@minValue(0)
@maxValue(3)
param minReplicas int = 0

var suffix = uniqueString(resourceGroup().id)
var tags = {
  project: 'llm-guardrail'
  purpose: 'serving'
}
var appName = '${prefix}-shadow'
var tokenSecretName = 'guardrail-api-token'

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: take('${prefix}acr${suffix}', 50)
  location: location
  tags: tags
  sku: { name: 'Basic' }
  // Basic tier: anonymous pull is unavailable (Standard and above only) and stays off.
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-pull-${suffix}'
  location: location
  tags: tags
}

// Built-in role "AcrPull": the app can only read images.
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource pullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, pullIdentity.id, acrPullRoleId)
  scope: registry
  properties: {
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
  }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-serving-logs-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env-${suffix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${pullIdentity.id}': {} }
  }
  dependsOn: [pullRole]
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        ipSecurityRestrictions: [
          for (range, index) in allowedSourceRanges: {
            name: 'allow-${index}'
            ipAddressRange: range
            action: 'Allow'
          }
        ]
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: pullIdentity.id
        }
      ]
      secrets: [
        {
          name: tokenSecretName
          value: apiToken
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'classifier'
          image: containerImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'GUARDRAIL_API_TOKEN', secretRef: tokenSecretName }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/readyz', port: 8000 }
              periodSeconds: 5
              failureThreshold: 36
            }
            {
              type: 'Readiness'
              httpGet: { path: '/readyz', port: 8000 }
              periodSeconds: 10
            }
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: 8000 }
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: 2
        rules: [
          {
            name: 'http'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

output registryName string = registry.name
output registryLoginServer string = registry.properties.loginServer
output appName string = appName
output appUrl string = deployApp ? 'https://${app!.properties.configuration.ingress.fqdn}' : ''
