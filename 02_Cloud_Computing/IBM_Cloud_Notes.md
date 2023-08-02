# IBM Cloud Notes

I made these notes while following the IBM/Coursera course [Introduction to Containers w/ Docker, Kubernetes & OpenShift](https://www.coursera.org/learn/ibm-containers-docker-kubernetes-openshift?specialization=devops-and-software-engineering), not the Udacity Computer Vision Nanodegree.

During the course, students got a free account for a given period.

## Table of Contents

- [IBM Cloud Notes](#ibm-cloud-notes)
  - [Table of Contents](#table-of-contents)
  - [1. Install and Set Up the IBM Cloud CLI](#1-install-and-set-up-the-ibm-cloud-cli)
  - [2. Create a Container Registry Namespace](#2-create-a-container-registry-namespace)

## 1. Install and Set Up the IBM Cloud CLI

First, create an account at IBM Cloud.

Then:

    Download installer and install the CLI tool
        https://github.com/IBM-Cloud/ibm-cloud-cli-release/releases/
    Log in to IBM Cloud (and select region, e.g., us-south)
        >> ibmcloud login # if it doesn't work, try the federated login below with --sso
        >> ibmcloud login --sso # for federated login, then follow web & iOS app
            IBMID: mxagar@gmail.com
            pw
    Install plugins
        IBM Cloud Kubernetes Service: ibmcloud ks
            >> ibmcloud plugin install container-service
        IBM Cloud Container Registry: ibmcloud cr
            >> ibmcloud plugin install container-registry
        IBM Cloud Monitoring: ibmcloud ob
            >> ibmcloud plugin install observe-service
        List plugins
            >> ibmcloud plugin list

Source: [Setting up the CLI](https://cloud.ibm.com/docs/containers?topic=containers-cs_cli_install)

Links with commands:

- [`ibmcloud`](https://cloud.ibm.com/docs/cli/reference/ibmcloud?topic=cli-ibmcloud_cli#ibmcloud_cli)
- [`ibmcloud ks`](https://cloud.ibm.com/docs/containers?topic=containers-kubernetes-service-cli)
- [`ibmcloud cr`](https://cloud.ibm.com/docs/Registry?topic=Registry-containerregcli)
- [`ibmcloud ob`](https://cloud.ibm.com/docs/containers?topic=containers-observability_cli)

## 2. Create a Container Registry Namespace

> The IBM Cloud Container Registry provides a multi-tenant, encrypted private image registry that you can use to store and access your container images in a highly available and scalable architecture. The namespace is a slice of the registry to which you can push your images. The namespace will be a part of the image name when you tag and push an image. For example, 
> `us.icr.io/<my_namespace>/<my_repo>:<my_tag>`

Steps to create a container registry namespace:

1. Log in to IBM Cloud: [https://cloud.ibm.com/](https://cloud.ibm.com/).
2. In the Catalogue, search *Container Registry*.
3. Click on *Get Started*.
4. Select location, e.g., Dallas.
5. Click on left menu panel: *Namespaces*
6. Create

        Resource: Default
        Name (must be unique): ibm_kubernetes_1

The [Quick start](https://cloud.ibm.com/registry/start) pannel on the left menu has a quick command guide to set up a namespace via the CLI.

