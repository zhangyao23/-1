#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI异常检测系统部署脚本

自动化部署脚本，用于在目标设备（如路由器、网络设备）上
部署和配置AI异常检测系统。

主要功能：
1. 系统环境检查和准备
2. 依赖库安装
3. 配置文件部署
4. 服务安装和启动
5. 系统测试和验证
"""

import os
import sys
import subprocess
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


class DeploymentManager:
    """
    部署管理器
    
    负责AI异常检测系统的完整部署流程，
    包括环境准备、软件安装、配置部署等。
    """
    
    def __init__(self, config_file="config/deployment_config.json"):
        """
        初始化部署管理器
        
        Args:
            config_file: 部署配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_deployment_config()
        
        # 部署路径设置
        self.source_dir = Path(__file__).parent.parent
        self.target_dir = Path(self.config.get('target_directory', '/opt/anomaly_detector'))
        self.service_name = self.config.get('service_name', 'anomaly-detector')
        
        # 日志设置
        self.log_file = Path(self.config.get('log_file', '/var/log/deployment.log'))
        
        print(f"部署管理器初始化完成")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
    
    def _load_deployment_config(self):
        """加载部署配置"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 使用默认配置
                return self._get_default_config()
        except Exception as e:
            print(f"加载部署配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认部署配置"""
        return {
            "target_directory": "/opt/anomaly_detector",
            "service_name": "anomaly-detector",
            "user": "anomaly",
            "group": "anomaly",
            "log_file": "/var/log/deployment.log",
            "python_version": "3.8",
            "create_virtual_env": True,
            "install_dependencies": True,
            "enable_service": True,
            "start_service": True,
            "backup_existing": True
        }
    
    def deploy(self, dry_run=False):
        """
        执行完整部署流程
        
        Args:
            dry_run: 是否为试运行模式
        """
        try:
            self._log("开始AI异常检测系统部署")
            
            # 1. 系统环境检查
            if not self._check_system_requirements():
                self._log("系统环境检查失败", level="ERROR")
                return False
            
            # 2. 创建用户和组
            if not dry_run:
                self._create_system_user()
            
            # 3. 备份现有安装
            if not dry_run and self.config.get('backup_existing', True):
                self._backup_existing_installation()
            
            # 4. 创建目标目录结构
            if not dry_run:
                self._create_directory_structure()
            
            # 5. 复制源文件
            if not dry_run:
                self._copy_source_files()
            
            # 6. 设置Python环境
            if not dry_run and self.config.get('create_virtual_env', True):
                self._setup_python_environment()
            
            # 7. 安装依赖
            if not dry_run and self.config.get('install_dependencies', True):
                self._install_dependencies()
            
            # 8. 部署配置文件
            if not dry_run:
                self._deploy_configuration()
            
            # 9. 设置权限
            if not dry_run:
                self._set_permissions()
            
            # 10. 创建系统服务
            if not dry_run:
                self._create_system_service()
            
            # 11. 启用并启动服务
            if not dry_run and self.config.get('enable_service', True):
                self._enable_and_start_service()
            
            # 12. 验证部署
            if not dry_run:
                self._verify_deployment()
            
            self._log("AI异常检测系统部署完成")
            return True
            
        except Exception as e:
            self._log(f"部署过程中发生错误: {e}", level="ERROR")
            return False
    
    def _check_system_requirements(self):
        """检查系统要求"""
        self._log("检查系统要求...")
        
        # 检查操作系统
        if not sys.platform.startswith('linux'):
            self._log("当前仅支持Linux系统", level="ERROR")
            return False
        
        # 检查Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_version = self.config.get('python_version', '3.8')
        
        if python_version < required_version:
            self._log(f"Python版本过低，需要 {required_version}+，当前: {python_version}", level="ERROR")
            return False
        
        # 检查磁盘空间
        disk_usage = shutil.disk_usage(self.target_dir.parent)
        free_space_mb = disk_usage.free / (1024 * 1024)
        
        if free_space_mb < 500:  # 需要至少500MB空间
            self._log(f"磁盘空间不足，需要至少500MB，当前可用: {free_space_mb:.1f}MB", level="ERROR")
            return False
        
        # 检查必要的系统命令
        required_commands = ['pip', 'systemctl']
        for cmd in required_commands:
            if not self._command_exists(cmd):
                self._log(f"缺少必要命令: {cmd}", level="ERROR")
                return False
        
        self._log("系统要求检查通过")
        return True
    
    def _command_exists(self, command):
        """检查命令是否存在"""
        try:
            subprocess.run(['which', command], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _create_system_user(self):
        """创建系统用户和组"""
        self._log("创建系统用户和组...")
        
        user = self.config.get('user', 'anomaly')
        group = self.config.get('group', 'anomaly')
        
        try:
            # 创建组
            subprocess.run(['groupadd', '-f', group], check=True)
            
            # 创建用户
            subprocess.run([
                'useradd', '-r', '-g', group, '-d', str(self.target_dir),
                '-s', '/bin/false', '-c', 'Anomaly Detector Service', user
            ], check=True)
            
            self._log(f"用户 {user} 和组 {group} 创建成功")
            
        except subprocess.CalledProcessError as e:
            if e.returncode == 9:  # 用户已存在
                self._log(f"用户 {user} 已存在")
            else:
                raise
    
    def _backup_existing_installation(self):
        """备份现有安装"""
        if self.target_dir.exists():
            backup_dir = Path(f"{self.target_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self._log(f"备份现有安装到: {backup_dir}")
            shutil.move(str(self.target_dir), str(backup_dir))
    
    def _create_directory_structure(self):
        """创建目录结构"""
        self._log("创建目录结构...")
        
        directories = [
            self.target_dir,
            self.target_dir / 'src',
            self.target_dir / 'config',
            self.target_dir / 'models',
            self.target_dir / 'data',
            self.target_dir / 'logs',
            self.target_dir / 'scripts'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self._log(f"创建目录: {directory}")
    
    def _copy_source_files(self):
        """复制源文件"""
        self._log("复制源文件...")
        
        # 复制源代码
        src_src = self.source_dir / 'src'
        if src_src.exists():
            shutil.copytree(src_src, self.target_dir / 'src', dirs_exist_ok=True)
        
        # 复制配置文件
        src_config = self.source_dir / 'config'
        if src_config.exists():
            shutil.copytree(src_config, self.target_dir / 'config', dirs_exist_ok=True)
        
        # 复制脚本
        src_scripts = self.source_dir / 'scripts'
        if src_scripts.exists():
            shutil.copytree(src_scripts, self.target_dir / 'scripts', dirs_exist_ok=True)
        
        # 复制requirements.txt
        requirements_file = self.source_dir / 'requirements.txt'
        if requirements_file.exists():
            shutil.copy2(requirements_file, self.target_dir / 'requirements.txt')
    
    def _setup_python_environment(self):
        """设置Python虚拟环境"""
        self._log("设置Python虚拟环境...")
        
        venv_path = self.target_dir / 'venv'
        
        # 创建虚拟环境
        subprocess.run([
            sys.executable, '-m', 'venv', str(venv_path)
        ], check=True)
        
        self._log(f"Python虚拟环境已创建: {venv_path}")
    
    def _install_dependencies(self):
        """安装Python依赖"""
        self._log("安装Python依赖...")
        
        pip_executable = self.target_dir / 'venv' / 'bin' / 'pip'
        requirements_file = self.target_dir / 'requirements.txt'
        
        if requirements_file.exists():
            subprocess.run([
                str(pip_executable), 'install', '-r', str(requirements_file)
            ], check=True)
            self._log("Python依赖安装完成")
        else:
            self._log("未找到requirements.txt文件", level="WARNING")
    
    def _deploy_configuration(self):
        """部署配置文件"""
        self._log("部署配置文件...")
        
        # 这里可以根据实际环境调整配置
        config_file = self.target_dir / 'config' / 'system_config.json'
        if config_file.exists():
            # 读取并修改配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 调整路径配置
            config['logging']['file_path'] = str(self.target_dir / 'logs' / 'anomaly_detector.log')
            config['storage']['anomaly_data_path'] = str(self.target_dir / 'data' / 'anomalies')
            config['ai_models']['autoencoder']['model_path'] = str(self.target_dir / 'models' / 'autoencoder.h5')
            config['ai_models']['classifier']['model_path'] = str(self.target_dir / 'models' / 'error_classifier.pkl')
            
            # 写回配置文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self._log("配置文件已更新")
    
    def _set_permissions(self):
        """设置文件权限"""
        self._log("设置文件权限...")
        
        user = self.config.get('user', 'anomaly')
        group = self.config.get('group', 'anomaly')
        
        # 设置目录所有者
        subprocess.run([
            'chown', '-R', f'{user}:{group}', str(self.target_dir)
        ], check=True)
        
        # 设置可执行权限
        main_script = self.target_dir / 'src' / 'main.py'
        if main_script.exists():
            subprocess.run(['chmod', '+x', str(main_script)], check=True)
    
    def _create_system_service(self):
        """创建systemd服务"""
        self._log("创建系统服务...")
        
        service_content = f"""[Unit]
Description=AI Network Anomaly Detector
After=network.target

[Service]
Type=simple
User={self.config.get('user', 'anomaly')}
Group={self.config.get('group', 'anomaly')}
WorkingDirectory={self.target_dir}
Environment=PATH={self.target_dir}/venv/bin
ExecStart={self.target_dir}/venv/bin/python {self.target_dir}/src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = Path(f'/etc/systemd/system/{self.service_name}.service')
        with open(service_file, 'w', encoding='utf-8') as f:
            f.write(service_content)
        
        # 重新加载systemd配置
        subprocess.run(['systemctl', 'daemon-reload'], check=True)
        self._log(f"系统服务 {self.service_name} 已创建")
    
    def _enable_and_start_service(self):
        """启用并启动服务"""
        self._log("启用并启动服务...")
        
        # 启用服务
        subprocess.run(['systemctl', 'enable', self.service_name], check=True)
        
        # 启动服务
        if self.config.get('start_service', True):
            subprocess.run(['systemctl', 'start', self.service_name], check=True)
        
        self._log(f"服务 {self.service_name} 已启用并启动")
    
    def _verify_deployment(self):
        """验证部署"""
        self._log("验证部署...")
        
        # 检查服务状态
        try:
            result = subprocess.run([
                'systemctl', 'is-active', self.service_name
            ], capture_output=True, text=True)
            
            if result.stdout.strip() == 'active':
                self._log("服务运行正常")
            else:
                self._log(f"服务状态异常: {result.stdout.strip()}", level="WARNING")
        
        except Exception as e:
            self._log(f"服务状态检查失败: {e}", level="ERROR")
        
        # 检查日志文件
        log_file = self.target_dir / 'logs' / 'anomaly_detector.log'
        if log_file.exists():
            self._log("日志文件存在")
        else:
            self._log("日志文件不存在", level="WARNING")
    
    def _log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        # 同时写入日志文件
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception:
            pass  # 忽略日志写入错误
    
    def uninstall(self):
        """卸载系统"""
        self._log("开始卸载AI异常检测系统")
        
        try:
            # 停止服务
            subprocess.run(['systemctl', 'stop', self.service_name], check=False)
            
            # 禁用服务
            subprocess.run(['systemctl', 'disable', self.service_name], check=False)
            
            # 删除服务文件
            service_file = Path(f'/etc/systemd/system/{self.service_name}.service')
            if service_file.exists():
                service_file.unlink()
                subprocess.run(['systemctl', 'daemon-reload'], check=True)
            
            # 删除安装目录
            if self.target_dir.exists():
                shutil.rmtree(self.target_dir)
            
            # 删除用户（可选）
            user = self.config.get('user', 'anomaly')
            subprocess.run(['userdel', user], check=False)
            
            self._log("卸载完成")
            return True
            
        except Exception as e:
            self._log(f"卸载过程中发生错误: {e}", level="ERROR")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI异常检测系统部署工具')
    parser.add_argument('action', choices=['deploy', 'uninstall'], help='执行的操作')
    parser.add_argument('--config', default='config/deployment_config.json', help='部署配置文件')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式（不实际执行操作）')
    
    args = parser.parse_args()
    
    # 检查权限
    if os.geteuid() != 0:
        print("错误: 需要root权限运行此脚本")
        sys.exit(1)
    
    try:
        manager = DeploymentManager(args.config)
        
        if args.action == 'deploy':
            success = manager.deploy(dry_run=args.dry_run)
        elif args.action == 'uninstall':
            success = manager.uninstall()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 